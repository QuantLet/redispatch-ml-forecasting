#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat <<'EOF'
Usage: delete_checkpoint_from_model_data.sh --root <dir> --mode <last|best|both>

Delete selected checkpoint files from .tar.zst model archives and re-pack each
archive using the same path and filename.

Options:
	--root <dir>            Root directory to scan recursively for .tar.zst archives.
	--mode <last|best|both> Which checkpoints to delete.
	-h, --help              Show this help.

Examples:
	delete_checkpoint_from_model_data.sh --root outputs --mode both
	delete_checkpoint_from_model_data.sh --root outputs_no_covariates --mode best
EOF
}

ROOT_DIR=""
MODE=""

while [[ $# -gt 0 ]]; do
	case "$1" in
		--root)
			ROOT_DIR="${2:-}"
			shift 2
			;;
		--mode)
			MODE="${2:-}"
			shift 2
			;;
		-h|--help)
			usage
			exit 0
			;;
		*)
			echo "Unknown argument: $1" >&2
			usage
			exit 1
			;;
	esac
done

if [[ -z "$ROOT_DIR" || -z "$MODE" ]]; then
	echo "Error: --root and --mode are required." >&2
	usage
	exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
	echo "Error: root directory does not exist: $ROOT_DIR" >&2
	exit 1
fi

case "$MODE" in
	last|best|both) ;;
	*)
		echo "Error: --mode must be one of: last, best, both" >&2
		exit 1
		;;
esac

if ! command -v tar >/dev/null 2>&1; then
	echo "Error: tar is required but not found." >&2
	exit 1
fi

if ! command -v unzstd >/dev/null 2>&1; then
	echo "Error: unzstd is required but not found." >&2
	exit 1
fi

if ! command -v zstd >/dev/null 2>&1; then
	echo "Error: zstd is required but not found." >&2
	exit 1
fi

count_archives=0
count_updated=0
count_skipped=0

while IFS= read -r -d '' archive; do
	((count_archives += 1))
	echo "Processing: $archive"

	tmpdir="$(mktemp -d)"
	tmp_archive="${archive}.tmp.$$"

	cleanup_tmp() {
		rm -rf "$tmpdir"
		rm -f "$tmp_archive"
	}

	if ! tar --use-compress-program=unzstd -xf "$archive" -C "$tmpdir"; then
		echo "  Failed to extract archive, skipping."
		((count_skipped += 1))
		cleanup_tmp
		continue
	fi

	deleted_any=0

	if [[ "$MODE" == "best" || "$MODE" == "both" ]]; then
		while IFS= read -r -d '' f; do
			rm -f "$f"
			deleted_any=1
			echo "  Deleted best checkpoint: ${f#$tmpdir/}"
		done < <(find "$tmpdir" -type f -name 'best_valid_checkpoint.ckpt' -print0)
	fi

	if [[ "$MODE" == "last" || "$MODE" == "both" ]]; then
		while IFS= read -r -d '' f; do
			rm -f "$f"
			deleted_any=1
			echo "  Deleted last checkpoint: ${f#$tmpdir/}"
		done < <(find "$tmpdir" -type f \( -name '*_0.ckpt' -o -name 'last.ckpt' -o -name 'last_checkpoint.ckpt' \) -print0)
	fi

	if [[ "$deleted_any" -eq 0 ]]; then
		echo "  No matching checkpoints found, archive unchanged."
		((count_skipped += 1))
		cleanup_tmp
		continue
	fi

	mapfile -d '' top_level_entries < <(find "$tmpdir" -mindepth 1 -maxdepth 1 -printf '%P\0')
	if [[ ${#top_level_entries[@]} -eq 0 ]]; then
		echo "  Extracted archive appears empty after cleanup, skipping."
		((count_skipped += 1))
		cleanup_tmp
		continue
	fi

	if ! tar --use-compress-program='zstd -T0' -cf "$tmp_archive" -C "$tmpdir" "${top_level_entries[@]}"; then
		echo "  Failed to re-pack archive, leaving original unchanged."
		((count_skipped += 1))
		cleanup_tmp
		continue
	fi

	mv "$tmp_archive" "$archive"
	((count_updated += 1))
	echo "  Updated archive in place."

	cleanup_tmp
done < <(find "$ROOT_DIR" -type f -name '*.tar.zst' -print0)

echo
echo "Done."
echo "  Archives scanned : $count_archives"
echo "  Archives updated : $count_updated"
echo "  Archives skipped : $count_skipped"
