#!/usr/bin/env bash
set -euo pipefail

function usage() {
  cat <<EOF
Usage:
  $0 split <input_file.onnx> <output_prefix>
    - Divide <input_file.onnx> in 3 parti: <output_prefix>_part1.onnx, _part2.onnx, _part3.onnx

  $0 merge <input_prefix> <output_file.onnx>
    - Ricompone in <output_file.onnx> i file <input_prefix>_part1.onnx, _part2.onnx, _part3.onnx

Examples:
  $0 split model.onnx model
  $0 merge model model_rebuilt.onnx
EOF
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

cmd="$1"; shift

case "$cmd" in

  split)
    if [[ $# -ne 2 ]]; then usage; fi
    input="$1"; prefix="$2"

    # Get file size (GNU or BSD stat)
    if stat --version >/dev/null 2>&1; then
      filesize=$(stat --printf="%s" "$input")
    else
      filesize=$(stat -f%z "$input")
    fi

    part_size=$((filesize / 3))
    part1_size=$part_size
    part2_size=$part_size
    part3_size=$((filesize - part1_size - part2_size))

    echo "Splitting '$input' (${filesize} bytes) into:"
    echo "  1) ${part1_size} bytes → ${prefix}_part1.onnx"
    echo "  2) ${part2_size} bytes → ${prefix}_part2.onnx"
    echo "  3) ${part3_size} bytes → ${prefix}_part3.onnx"

    head -c "$part1_size" "$input" > "${prefix}_part1.onnx"
    head -c $((part1_size + part2_size)) "$input" | tail -c "$part2_size" > "${prefix}_part2.onnx"
    tail -c "$part3_size" "$input" > "${prefix}_part3.onnx"

    echo "Split completato."
    ;;

  merge)
    if [[ $# -ne 2 ]]; then usage; fi
    prefix="$1"; output="$2"

    part1="${prefix}_part1.onnx"
    part2="${prefix}_part2.onnx"
    part3="${prefix}_part3.onnx"

    for f in "$part1" "$part2" "$part3"; do
      if [[ ! -f "$f" ]]; then
        echo "Errore: file mancante '$f'" >&2
        exit 2
      fi
    done

    echo "Merging:"
    echo "  1) $part1"
    echo "  2) $part2"
    echo "  3) $part3"
    echo "→ $output"

    # Concatenate in ordine
    cat "$part1" "$part2" "$part3" > "$output"

    echo "Merge completato."
    ;;

  *)
    echo "Comando sconosciuto: $cmd" >&2
    usage
    ;;
esac
