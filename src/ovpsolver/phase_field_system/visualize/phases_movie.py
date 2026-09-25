"""Consecutive saved frames of a run, rendered and encoded as an MP4 or a GIF."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import threading
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from ...fem.save import Run
from ..entry import EntryPoint
from ..run import read
from .style import draw_without_a_window

if TYPE_CHECKING:
    from argparse import ArgumentParser
    from collections.abc import Sequence

    from ..system import PhaseFieldSystem

logger = logging.getLogger(__name__)

#: Frame delays in a GIF are stored in centiseconds, so only rates dividing 100
#: play at the rate asked for: 12 fps becomes a 9 cs delay and plays at 11.1.
GIF_EXACT_FPS = (1, 2, 4, 5, 10, 20, 25, 50, 100)

#: The rendered frames' file names, numbered from zero, as ffmpeg reads them.
FRAME_PATTERN = "frame-%09d.png"


def movie(
    system_class: type[PhaseFieldSystem],
    spec: str | Path,
    *,
    frames: Sequence[int],
    out: str | Path | None = None,
    fps: int = 12,
    quality: int = 100,
    gif: bool = False,
    keep_cache: bool = True,
    split_fields: bool = False,
    group_polymerizing: bool = False,
    diffuse_domain: bool = True,
    with_ddm_floor: bool = False,
    joint_colorbars: bool = False,
    aux_fields: Sequence[str] = (),
    aux_cmap: str | None = None,
    split_titles: Sequence[str] | None = None,
    n_rows: int = 1,
    **style_overrides: Any,
) -> Path:
    """Render frames ``START`` to ``END`` inclusive, encode them, return the path.

    ``quality`` runs from 0 to 100: the CRF for H.264, and for ``gif`` the
    palette size, since a GIF quantizes the whole clip to one palette. ``gif``
    writes ``movie.gif`` instead of ``movie.mp4``, for somewhere that will not
    play video; expect it an order of magnitude larger (on gradient content at
    320x240, H.264 at CRF 23 is 28 kB and the best GIF 104 kB).

    The rendered frames are kept beside the output, in a directory named by a
    hash of the spec and of every setting that decides what they look like, so
    a second call differing only in ``fps``, ``quality`` or ``gif`` redraws
    nothing: rendering is the slow half. Clearing ``keep_cache`` renders into a
    temporary directory instead.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "phases_movie needs ffmpeg on PATH; install it in the active environment"
        )
    pair = tuple(int(frame) for frame in frames)
    if len(pair) != 2:
        raise ValueError(f"movie needs exactly START and END frames, got {len(pair)}")
    start, end = pair
    if start < 0 or end < 0:
        raise ValueError("frame indices are non-negative")
    if start > end:
        raise ValueError(f"START frame {start} follows END frame {end}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if gif and fps not in GIF_EXACT_FPS:
        nearest = min(GIF_EXACT_FPS, key=lambda one: abs(one - fps))
        logger.warning(
            "a GIF stores frame delays in centiseconds, so %d fps is rounded to "
            "%.2f fps on playback; --fps %d plays at the rate asked for",
            fps, 100.0 / round(100.0 / fps), nearest,
        )
    if not 0 <= quality <= 100:
        raise ValueError("quality must be between 0 and 100")

    spec = Path(spec)
    parameters = read(system_class.Parameters, spec)
    run = Run.of(parameters)
    if end >= run.n_frames:
        raise ValueError(
            f"frame {end} is outside the {run.n_frames} frame(s) saved in {run.path}"
        )
    suffix = "gif" if gif else "mp4"
    output = run.path / f"movie.{suffix}" if out is None else Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Everything that decides what the frames look like, and nothing that
    # decides how they are encoded: the split is what the cache keys on.
    frame_options = dict(
        split_fields=split_fields,
        group_polymerizing=group_polymerizing,
        diffuse_domain=diffuse_domain,
        with_ddm_floor=with_ddm_floor,
        joint_colorbars=joint_colorbars,
        aux_fields=tuple(aux_fields),
        aux_cmap=aux_cmap,
        split_titles=None if split_titles is None else tuple(split_titles),
        n_rows=n_rows,
        **style_overrides,
    )
    names = [FRAME_PATTERN % number for number in range(end - start + 1)]

    draw_without_a_window()
    with ExitStack() as stack:
        if keep_cache:
            key = _fingerprint(system_class, spec, run, start, end, frame_options)
            temporary = output.parent / f".ovpsolver-movie-{key}"
            temporary.mkdir(parents=True, exist_ok=True)
        else:
            temporary = Path(
                stack.enter_context(
                    TemporaryDirectory(prefix=".ovpsolver-movie-", dir=output.parent)
                )
            )

        # Printed rather than logged: the CLI installs no logging configuration,
        # so an info record would go nowhere, and a reused cache is the
        # difference between a picture of this run and one of whatever was
        # rendered last.
        if all((temporary / name).is_file() for name in names):
            print(f"READING FROM CACHE: {temporary}")
            print(
                f"  {len(names)} frame(s) reused, nothing redrawn; only the "
                f"encoding settings take effect. Delete that directory to "
                f"force a redraw."
            )
        else:
            _render_frames(system_class, spec, temporary, start, end, frame_options)
            if keep_cache:
                print(f"WROTE CACHE: {temporary}")
                print(
                    f"  {len(names)} frame(s) kept for reuse. Delete that "
                    f"directory when you are done with it."
                )

        encoded = temporary / f"movie.{suffix}"
        encode = _encode_gif if gif else _encode_mp4
        encode(
            ffmpeg,
            temporary / FRAME_PATTERN,
            encoded,
            fps=fps,
            quality=quality,
            total=len(names),
        )
        encoded.replace(output)
    return output


def _fingerprint(
    system_class: type[PhaseFieldSystem],
    spec: Path,
    run: Run,
    start: int,
    end: int,
    options: dict[str, Any],
) -> str:
    """A name for the frames these settings produce, and only these.

    Over the mixture, the spec's text, the frame range and every option that
    reaches :meth:`PhaseFieldSystem.plot`, and none of the encoding settings.
    The run's ``meta.json`` is stamped in by size and modification time: running
    the spec again rewrites it, and a cache from the previous run would
    otherwise draw data that has since changed.
    """

    digest = hashlib.blake2b(digest_size=8)
    digest.update(f"{system_class.__module__}.{system_class.__name__}".encode())
    digest.update(spec.resolve().read_bytes())
    digest.update(f"{start}:{end}".encode())
    for name, value in sorted(options.items()):
        digest.update(f"{name}={value!r}".encode())
    meta = run.path / "meta.json"
    if meta.is_file():
        stat = meta.stat()
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _progress(**options: Any) -> tqdm:
    """A tqdm bar whose write lock is a threading lock.

    tqdm's default is a multiprocessing lock, which costs a named semaphore
    registered with the multiprocessing resource tracker: a tracker process that
    outlives :func:`~ovpsolver.phase_field_system.entry.exit_without_teardown`
    and then reports the semaphore as leaked. One process needs no more than a
    threading lock. Set before the bar is built, since tqdm makes its default on
    first use.
    """

    tqdm.set_lock(threading.RLock())
    return tqdm(**options)


def _render_frames(
    system_class: type[PhaseFieldSystem],
    spec: Path,
    directory: Path,
    start: int,
    end: int,
    options: dict[str, Any],
) -> None:
    """Draw each frame of the range into ``directory`` as a PNG, all one size.

    Lossless, since these are the encoder's input: a lossy intermediate would
    hand the encoder its own artefacts to spend bits on.
    """

    expected: tuple[int, ...] | None = None
    for number, frame in enumerate(
        _progress(iterable=range(start, end + 1), desc="rendering", unit="frame")
    ):
        figure = system_class.plot(spec, frames=(frame,), transpose=True, **options)
        dimensions = tuple(int(value) for value in figure.canvas.get_width_height())
        if expected is not None and dimensions != expected:
            plt.close(figure)
            raise RuntimeError(
                f"movie frame dimensions changed from {expected} to {dimensions}"
            )
        expected = dimensions
        figure.savefig(
            directory / (FRAME_PATTERN % number),
            dpi=figure.dpi,
            facecolor=figure.get_facecolor(),
            transparent=False,
        )
        plt.close(figure)


def _encode_mp4(
    ffmpeg: str,
    frames: Path,
    output: Path,
    *,
    fps: int,
    quality: int,
    total: int,
) -> None:
    # CRF 0 makes x264 select its poorly supported High 4:4:4 Predictive
    # profile even for 4:2:0 input. CRF 1 is visually lossless and stays in
    # the ordinary High profile understood by macOS and browser decoders.
    crf = max(1, round(51 * (100 - quality) / 100))
    arguments = [
        "-framerate",
        str(fps),
        "-i",
        str(frames),
        "-an",
        "-vf",
        (
            "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white,"
            "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
            "format=yuv420p"
        ),
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-x264-params",
        "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        "-y",
        str(output),
    ]
    _run(ffmpeg, arguments, total=total, description="encoding")


def _encode_gif(
    ffmpeg: str, frames: Path, output: Path, *, fps: int, quality: int, total: int
) -> None:
    """Encode the frames as an animated GIF, in two passes.

    A GIF carries one palette at 8 bits per pixel, so the clip is quantized as a
    whole: ``palettegen`` reads every frame and ``paletteuse`` maps them onto
    what it found, where a single pass takes a fixed 216-colour web palette.
    ``quality`` picks the palette size, from 32 to 256 colours.

    Dithering is ordered (``bayer``) rather than error-diffused, which is the
    largest lever on the file: error diffusion re-randomizes its noise every
    frame, leaving the GIF's compression -- LZW over runs, and dropping
    unchanged pixels -- nothing to work with. On gradient content, ``sierra2_4a``
    is 928 kB against ``bayer``'s 396 kB for the same palette.
    """

    colors = round(32 + (256 - 32) * quality / 100)
    palette = output.with_name("palette.png")
    dither = "bayer:bayer_scale=5"
    # The palette pass reads every frame but writes one, and ffmpeg counts what
    # it writes, so there is no total to measure that pass against.
    _run(
        ffmpeg,
        ["-framerate", str(fps), "-i", str(frames),
         "-vf", f"palettegen=max_colors={colors}:stats_mode=diff",
         "-y", str(palette)],
        total=None,
        description="reading palette",
    )
    _run(
        ffmpeg,
        ["-framerate", str(fps), "-i", str(frames), "-i", str(palette),
         "-lavfi", f"paletteuse=dither={dither}",
         "-loop", "0", "-f", "gif", "-y", str(output)],
        total=total,
        description="encoding gif",
    )


def _run(
    ffmpeg: str,
    arguments: list[str],
    *,
    total: int | None,
    description: str,
) -> None:
    """One ffmpeg invocation, with a progress bar, raising with its stderr on failure.

    Encoding a long run takes minutes with nothing to show for it, which looks
    like being stuck. ``-progress pipe:1`` makes ffmpeg report ``key=value``
    lines as it works, of which ``frame`` counts the frames written;
    ``-nostats`` turns off the status line it would otherwise redraw on stderr.

    stderr goes to a file rather than a pipe: two pipes read in sequence
    deadlock once the unread one fills its buffer, which happens exactly when a
    diagnostic is long, so when something has already gone wrong.
    """

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-progress",
        "pipe:1",
        "-nostats",
        *arguments,
    ]
    with TemporaryFile("w+") as errors:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=errors, text=True
        ) as process:
            with _progress(
                total=total, desc=description, unit="frame"
            ) as progress:
                for line in process.stdout:
                    key, _, value = line.strip().partition("=")
                    # ffmpeg reports a running total; tqdm wants the increment.
                    if key == "frame" and value.isdigit():
                        progress.update(int(value) - progress.n)
                if total is not None:
                    progress.update(total - progress.n)
        if process.returncode != 0:
            errors.seek(0)
            message = (
                errors.read().strip()
                or f"ffmpeg exited with status {process.returncode}"
            )
            raise RuntimeError(f"ffmpeg could not encode the movie: {message}")


class VisualizePhasesMovie(EntryPoint):
    """Render one run from START through END as an MP4 movie, or a GIF."""

    name = "visualize.phases_movie"
    summary = "render saved phase fields as an MP4 movie"

    #: The movie is on disk by the time this returns, and the exit hooks
    #: dolfinx installed are between there and the prompt.
    exits_without_teardown = True

    ## Overrides

    def declare(self, parser: ArgumentParser) -> None:
        parser.add_argument("spec", type=Path, help="path to the experiment YAML")
        parser.add_argument(
            "frames",
            nargs=2,
            type=int,
            metavar=("START", "END"),
            help="inclusive range of saved frame indices",
        )
        parser.add_argument(
            "--out",
            type=Path,
            help="output path; by default, movie.mp4 in the run's output folder",
        )
        parser.add_argument(
            "--fps",
            type=int,
            default=12,
            help="movie frames per second (default: 12)",
        )
        parser.add_argument(
            "--quality",
            type=int,
            default=100,
            metavar="0..100",
            help="encoding quality: the H.264 CRF, or the GIF palette size; "
            "lower values make smaller files (default: 100)",
        )
        parser.add_argument(
            "--delete-cache",
            dest="keep_cache",
            action="store_false",
            help="render into a temporary directory rather than keeping the "
            "frames beside the output; they are kept by default, named for the "
            "spec and the settings that decide what they look like, so "
            "re-encoding at a different --fps, --quality or --gif redraws "
            "nothing",
        )
        parser.add_argument(
            "--gif",
            action="store_true",
            help="write movie.gif instead of movie.mp4, for somewhere that will "
            "not play video; much larger for the same frames",
        )
        parser.add_argument(
            "--split-fields",
            "--split",
            dest="split_fields",
            action="store_true",
            help="give every phase its own panel beside the composite",
        )
        parser.add_argument(
            "--group-polymerizing",
            "--group-poly",
            dest="group_polymerizing",
            action="store_true",
            help="draw a crosslinking phase as one layer rather than as sol and gel",
        )
        parser.add_argument(
            "--no-diffuse-domain",
            dest="diffuse_domain",
            action="store_false",
            help=(
                "leave the fixed inclusions out, and draw every field unweighted by "
                "them: phases not multiplied by chi, ratios not masked inside the "
                "inclusions. A field saved already weighted is drawn as saved."
            ),
        )
        parser.add_argument(
            "--with-ddm-floor",
            action="store_true",
            help=(
                "weight fields by the indicator the solve carried, including "
                "indicator_floor, instead of by the indicator of the domain. Shows "
                "what the scheme did inside the inclusions, which is regularization "
                "rather than physics; off by default, when phases vanish there"
            ),
        )
        parser.add_argument(
            "--show-joint-colorbars",
            dest="joint_colorbars",
            action="store_true",
            help="add the composite phase colourbars",
        )
        parser.add_argument(
            "--aux-fields",
            nargs="+",
            default=(),
            dest="aux_fields",
            metavar="FIELD",
            help="saved fields to add as diagnostic panels",
        )
        parser.add_argument(
            "--aux-cmap",
            dest="aux_cmap",
            metavar="NAME",
            help="a colorcet colormap for the diagnostic panels, e.g. isolum",
        )
        parser.add_argument(
            "--n-rows",
            "--n_rows",
            type=int,
            default=1,
            dest="n_rows",
            metavar="N",
            help="fold the frame's strip of panels into N rows, filled left to "
            "right with the composite first and the last row ragged; for a frame "
            "naming enough fields to be unreadably wide as one strip. Needs "
            "panels to fold, so --split-fields or --aux-fields",
        )
        parser.add_argument(
            "--panel-pixels",
            type=int,
            dest="panel_pixels",
            help="the height of each rendered panel, in pixels",
        )
        parser.add_argument(
            "--font-scaling",
            type=float,
            dest="font_scaling",
            metavar="SCALE",
            help="multiply every font size on the movie frame by this, to read type "
            "that is too small or too large for the size it is shown at",
        )
        parser.add_argument("--dpi", type=float, help="resolution of each movie frame")
        parser.add_argument(
            "--split-titles",
            nargs="+",
            dest="split_titles",
            help="a title per detail panel, replacing the layer names",
        )

    def prepare(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # Left off the command line means the mixture's own plot_params value.
        for key in ("panel_pixels", "dpi", "font_scaling"):
            if arguments.get(key) is None:
                arguments.pop(key, None)
        return arguments

    def __call__(
        self, system_class: type[PhaseFieldSystem], *, spec: Path, **options: Any
    ) -> Path:
        return movie(system_class, spec, **options)
