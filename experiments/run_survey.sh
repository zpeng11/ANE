#!/bin/bash
# run_survey.sh — run every opcases/*.mil case and write exactly one CSV row per case.
#
# Usage:
#   ./run_survey.sh                       # fresh results/run-<UTC>.XXXXXXXX/
#   ./run_survey.sh results/my-run        # CSV + logs in results/my-run/
#   ./run_survey.sh /tmp/out.csv          # CSV there; fresh results run dir
#   ./run_survey.sh --csv FILE [--results DIR]
#   ./run_survey.sh --probe PATH          # default: build/op_probe (or $OP_PROBE)
#   ./run_survey.sh --list                # case table as NOT_RUN (CSV only)
#   ./run_survey.sh --layouts compact,align64,align128,align256
#   ./run_survey.sh --dump                # dump numeric-diagnosis surfaces
#   ./run_survey.sh --allow-unverified    # allow UNVERIFIED/LAYOUT_CANDIDATE
# --results DIR puts logs and metadata in DIR; the CSV defaults to
# DIR/survey_results.csv unless --csv says otherwise.  The default is a new,
# uniquely named results/run-<UTC>.XXXXXXXX directory.
#
# Existing CSV files and non-empty results directories are never overwritten:
# the script refuses with exit 2.  Use a new --output path for a new run.
#
# Before running (and in --list mode) opcases/_index.txt must match the MIL
# files exactly, and every indexed case must have a well-formed .meta and its
# weights.  A missing, duplicate or extra case is an error, not a silent pass.
#
# The CSV is written atomically and has the columns
#   op,status,compile,load,eval,numeric,exit
# where status is OK, COMPILE_FAIL, LOAD_FAIL, EVAL_FAIL, NUMERIC_FAIL,
# NUMERIC_LAYOUT_CANDIDATE, NUMERIC_UNVERIFIED, DESC_NIL, NOMIL, NOWEIGHTS,
# IOSURFACE_FAIL or RUN_ERROR.
# Multi-line NSError text goes only to survey_errors.log; it never leaks into
# the CSV.  Per-case stdout (including MAXDIFF/LAYOUT/BEST) is saved under
# <results>/stdout/<case>.log and in <results>/survey_stdout.log.
# A probe status that disagrees with its exit code (OK+nonzero,
# COMPILE_FAIL!=10, etc.) is recorded as RUN_ERROR, not as a pass.
#
# --list only writes the CSV: it does not create a results tree, stdout logs,
# survey_errors.log, run_meta.txt or summary.txt, and it never runs the probe.
#
# Exit status: 0 when every case is OK or an exploratory COMPILE_FAIL, or
# (with --allow-unverified) NUMERIC_UNVERIFIED / NUMERIC_LAYOUT_CANDIDATE.
# 1 when any case is NUMERIC_FAIL, LOAD_FAIL, NUMERIC_LAYOUT_CANDIDATE,
# NUMERIC_UNVERIFIED by default, or has a runtime/infra failure (EVAL_FAIL,
# RUN_ERROR, NOMIL, NOWEIGHTS, DESC_NIL, IOSURFACE_FAIL).  2 for usage errors.
#
# This script never deletes case files or unrelated directories.  It creates a
# fresh results directory by default and refuses to overwrite an existing CSV
# or a non-empty results directory.
set -u

cd "$(dirname "$0")" || exit 1

CSV=""
RESULTS=""
CSV_SET=0
RESULTS_SET=0
LIST=0
DUMP=0
ALLOW_UNVERIFIED=0
LAYOUTS=""
PROBE_OVERRIDE=""
POS=""

usage() {
    cat <<'EOF'
Usage:
  ./run_survey.sh                       CSV + logs in results/run/
  ./run_survey.sh results/my-run        CSV + logs in results/my-run/
  ./run_survey.sh /tmp/out.csv          CSV there; logs in results/run/
  ./run_survey.sh --csv FILE [--results DIR]
  ./run_survey.sh --probe PATH          default build/op_probe or $OP_PROBE
  ./run_survey.sh --list                case table as NOT_RUN (CSV only)
  ./run_survey.sh --layouts LIST        e.g. compact,align64,align128,align256
  ./run_survey.sh --dump                dump numeric-diagnosis surfaces
  ./run_survey.sh --allow-unverified    allow UNVERIFIED/LAYOUT_CANDIDATE
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv)
            [[ $# -ge 2 ]] || { echo "run_survey: --csv needs an argument" >&2; exit 2; }
            CSV="$2"; CSV_SET=1; shift 2 ;;
        --results)
            [[ $# -ge 2 ]] || { echo "run_survey: --results needs an argument" >&2; exit 2; }
            RESULTS="$2"; RESULTS_SET=1; shift 2 ;;
        --probe)
            [[ $# -ge 2 ]] || { echo "run_survey: --probe needs an argument" >&2; exit 2; }
            PROBE_OVERRIDE="$2"; shift 2 ;;
        --layouts)
            [[ $# -ge 2 ]] || { echo "run_survey: --layouts needs an argument" >&2; exit 2; }
            LAYOUTS="$2"; shift 2 ;;
        --dump)
            DUMP=1; shift ;;
        --allow-unverified)
            ALLOW_UNVERIFIED=1; shift ;;
        --strict)  # kept as an accepted no-op: strict is the default
            shift ;;
        --list)
            LIST=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        -*)
            echo "run_survey: unknown option $1" >&2; usage >&2; exit 2 ;;
        *)
            if [[ -n "$POS" ]]; then
                echo "run_survey: at most one positional argument" >&2; exit 2
            fi
            POS="$1"; shift ;;
    esac
done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [[ -n "$POS" ]]; then
    case "$POS" in
        *.csv)
            if [[ $CSV_SET -eq 0 ]]; then CSV="$POS"; CSV_SET=1; fi ;;
        *)
            if [[ $RESULTS_SET -eq 0 ]]; then
                RESULTS="$POS"; RESULTS_SET=1
            fi ;;
    esac
fi

# Never overwrite an existing CSV or reuse a non-empty results directory.
if [[ -n "$CSV" && -e "$CSV" ]]; then
    echo "run_survey: refusing to overwrite existing CSV: $CSV" >&2
    exit 2
fi
if [[ $RESULTS_SET -eq 1 ]]; then
    if [[ -e "$RESULTS" && ! -d "$RESULTS" ]]; then
        echo "run_survey: results path is not a directory: $RESULTS" >&2
        exit 2
    fi
    if [[ -d "$RESULTS" && -n "$(ls -A "$RESULTS" 2>/dev/null)" ]]; then
        echo "run_survey: refusing to reuse non-empty results directory: $RESULTS" >&2
        exit 2
    fi
fi

BUILD_DIR=${BUILD_DIR:-build}
PROBE=${PROBE_OVERRIDE:-${OP_PROBE:-$BUILD_DIR/op_probe}}
if [[ $LIST -eq 0 && ! -x "$PROBE" ]]; then
    echo "run_survey: $PROBE not found or not executable (run make all)" >&2
    exit 1
fi
if ! compgen -G "opcases/*.mil" > /dev/null; then
    echo "run_survey: no opcases/*.mil (run python3 mil_gen.py)" >&2
    exit 1
fi

preflight_manifest() {
    local index="opcases/_index.txt"
    local sorted actual dup missing extra name meta label inB outB wf
    if [[ ! -f "$index" ]]; then
        echo "run_survey: missing case manifest: $index" >&2
        return 1
    fi
    sorted=$(mktemp) || return 1
    actual=$(mktemp) || { rm -f "$sorted"; return 1; }
    sed '/^[[:space:]]*$/d' "$index" | LC_ALL=C sort > "$sorted"
    if [[ ! -s "$sorted" ]]; then
        echo "run_survey: empty case manifest: $index" >&2
        rm -f "$sorted" "$actual"; return 1
    fi
    dup=$(uniq -d "$sorted")
    if [[ -n "$dup" ]]; then
        echo "run_survey: duplicate case names in $index: $(printf '%s' "$dup" | tr '\n' ' ')" >&2
        rm -f "$sorted" "$actual"; return 1
    fi
    for f in opcases/*.mil; do
        [[ -e "$f" ]] || continue
        basename "$f" .mil
    done | LC_ALL=C sort > "$actual"
    if ! cmp -s "$sorted" "$actual"; then
        missing=$(comm -23 "$sorted" "$actual" | tr '\n' ' ')
        extra=$(comm -13 "$sorted" "$actual" | tr '\n' ' ')
        echo "run_survey: case manifest mismatch (missing MIL: ${missing:-none}; extra MIL: ${extra:-none})" >&2
        rm -f "$sorted" "$actual"; return 1
    fi
    while IFS= read -r name; do
        meta="opcases/$name.meta"
        if [[ ! -f "$meta" ]]; then
            echo "run_survey: missing meta file: $meta" >&2
            rm -f "$sorted" "$actual"; return 1
        fi
        IFS=, read -r label inB outB wf < "$meta"
        if [[ "$label" != "$name" ]]; then
            echo "run_survey: meta label mismatch in $meta: '$label' != '$name'" >&2
            rm -f "$sorted" "$actual"; return 1
        fi
        case "$inB" in
            ''|*[!0-9]*) echo "run_survey: invalid input byte count in $meta" >&2
                          rm -f "$sorted" "$actual"; return 1 ;;
        esac
        case "$outB" in
            ''|*[!0-9]*) echo "run_survey: invalid output byte count in $meta" >&2
                          rm -f "$sorted" "$actual"; return 1 ;;
        esac
        if [[ "$inB" -le 0 || "$outB" -le 0 ]]; then
            echo "run_survey: non-positive byte count in $meta" >&2
            rm -f "$sorted" "$actual"; return 1
        fi
        case "$wf" in
            -|"") : ;;
            /*|*..*) echo "run_survey: invalid weights path in $meta: $wf" >&2
                     rm -f "$sorted" "$actual"; return 1 ;;
            *) if [[ ! -f "$wf" ]]; then
                   echo "run_survey: missing weights file for $name: $wf" >&2
                   rm -f "$sorted" "$actual"; return 1
               fi ;;
        esac
    done < "$sorted"
    rm -f "$sorted" "$actual"
    return 0
}

if ! preflight_manifest; then
    exit 1
fi

# A hardware run always needs a logs directory.  --list needs one only when it
# has to derive the default CSV location.  The default is always a brand-new
# uniquely named directory.
if [[ -z "$RESULTS" && ( -z "$CSV" || $LIST -eq 0 ) ]]; then
    mkdir -p results || exit 1
    RESULTS=$(mktemp -d "results/run-${STAMP}.XXXXXXXX") || exit 1
fi
if [[ -n "$RESULTS" ]]; then
    mkdir -p "$RESULTS" || exit 1
fi
if [[ -z "$CSV" ]]; then
    CSV="$RESULTS/survey_results.csv"
fi
if [[ -e "$CSV" ]]; then
    echo "run_survey: refusing to overwrite existing CSV: $CSV" >&2
    exit 2
fi

mkdir -p "$(dirname "$CSV")" || exit 1
TMP=$(mktemp "$(dirname "$CSV")/.survey_results.XXXXXX") || exit 1
trap 'rm -f "$TMP"' EXIT
printf 'op,status,compile,load,eval,numeric,exit\n' > "$TMP"

RAW=""
COMBINED=""
STDOUT_DIR=""
if [[ $LIST -eq 0 ]]; then
    mkdir -p "$RESULTS" || exit 1
    STDOUT_DIR="$RESULTS/stdout"
    mkdir -p "$STDOUT_DIR" || exit 1
    RAW="$RESULTS/survey_errors.log"
    COMBINED="$RESULTS/survey_stdout.log"
    : > "$RAW"
    : > "$COMBINED"
fi

if [[ -n "$LAYOUTS" ]]; then export OP_PROBE_LAYOUTS="$LAYOUTS"; fi
if [[ $DUMP -eq 1 && $LIST -eq 0 ]]; then
    mkdir -p "$RESULTS/dumps" || exit 1
    export OP_PROBE_DUMP_DIR="$RESULTS/dumps"
    export OP_PROBE_DUMP_ATTRS=1
fi

# Generous fixed I/O surfaces (the prior survey used 256 KiB to avoid the
# small-surface eval failure).  Numeric checks use the shapes parsed from MIL.
SURFACE=262144
n=0
numeric_fail=0
numeric_unverified=0
layout_candidate=0
infra_fail=0

field() {
    printf '%s\n' "$1" | sed -n "s/.* $2=\([^ ]*\).*/\1/p" | head -n 1
}

record_stdout() {
    printf '%s\n' "$2" > "$STDOUT_DIR/$1.log"
    printf '\n--- %s ---\n%s\n' "$1" "$2" >> "$COMBINED"
}

# Probe exit code required for each status.  A mismatch is an infra failure,
# not a pass (in particular OK+nonzero and COMPILE_FAIL != 10 are rejected).
expected_rc() {
    case "$1" in
        OK) echo 0 ;;
        COMPILE_FAIL) echo 10 ;;
        LOAD_FAIL) echo 11 ;;
        EVAL_FAIL) echo 12 ;;
        NUMERIC_FAIL) echo 13 ;;
        NUMERIC_UNVERIFIED) echo 0 ;;
        NUMERIC_LAYOUT_CANDIDATE) echo 0 ;;
        DESC_NIL) echo 4 ;;
        NOMIL|NOWEIGHTS) echo 3 ;;
        IOSURFACE_FAIL) echo 14 ;;
        RUN_ERROR) echo 15 ;;
        *) echo -1 ;;
    esac
}

# Record paths without leaking the reviewer's home or checkout root.
portable() {
    local p=$1
    case "$p" in
        "$PWD"/*) p=${p#"$PWD"/} ;;
        "$HOME"/*) p="~/${p#"$HOME"/}" ;;
    esac
    printf '%s' "$p"
}

for f in opcases/*.mil; do
    [[ -e "$f" ]] || continue
    n=$((n + 1))
    base=$(basename "$f" .mil)
    meta="opcases/$base.meta"

    if [[ ! -f "$meta" ]]; then
        printf '%s,RUN_ERROR,NA,NA,NA,NA,-1\n' "${base//,/_;}" >> "$TMP"
        printf 'run_survey: %s: missing meta file: %s\n' "$base" "$meta" >&2
        if [[ $LIST -eq 0 ]]; then
            printf '\n--- %s ---\nmissing meta file: %s\n' "$base" "$meta" >> "$RAW"
            record_stdout "$base" "RUN_ERROR missing-meta $meta"
        fi
        infra_fail=$((infra_fail + 1))
        continue
    fi

    IFS=, read -r label inB outB wf < "$meta"
    label=${label:-$base}
    if [[ -z "${wf:-}" ]]; then wf="-"; fi
    warg="$wf"
    if [[ "$wf" != "-" && ! -f "$wf" ]]; then
        printf '%s,RUN_ERROR,NA,NA,NA,NA,-1\n' "${label//,/_;}" >> "$TMP"
        printf 'run_survey: %s: missing weights file: %s\n' "$base" "$wf" >&2
        if [[ $LIST -eq 0 ]]; then
            printf '\n--- %s ---\nmissing weights file: %s\n' "$base" "$wf" >> "$RAW"
            record_stdout "$base" "RUN_ERROR missing-weights $wf"
        fi
        infra_fail=$((infra_fail + 1))
        continue
    fi

    if [[ $LIST -eq 1 ]]; then
        printf '%s,NOT_RUN,NA,NA,NA,NA,-1\n' "${label//,/_;}" >> "$TMP"
        continue
    fi

    printf '\n--- %s ---\n' "$base" >> "$RAW"
    line=$("$PROBE" "$f" "$warg" "$SURFACE" "$SURFACE" "$label" 2>>"$RAW")
    rc=$?
    line=$(printf '%s\n' "$line" | head -n 1)
    record_stdout "$base" "$line"

    status=$(printf '%s\n' "$line" | sed -n 's/.* STATUS \([^ ]*\).*/\1/p' | head -n 1)
    if [[ -z "$status" ]]; then
        status=RUN_ERROR
        c=NA; l=NA; e=NA; nu=NA
    else
        c=$(field "$line" COMPILE);   c=${c:-NA}
        l=$(field "$line" LOAD);      l=${l:-NA}
        e=$(field "$line" EVAL);      e=${e:-NA}
        nu=$(field "$line" NUMERIC);  nu=${nu:-NA}
    fi
    if [[ "$status" != "RUN_ERROR" ]]; then
        erc=$(expected_rc "$status")
        if [[ "$erc" == "-1" || "$rc" != "$erc" ]]; then
            printf '%s rc/status mismatch: status=%s rc=%s expected=%s -> RUN_ERROR\n' \
                "$base" "$status" "$rc" "$erc" >> "$RAW"
            status=RUN_ERROR
        fi
    fi
    printf '%s,%s,%s,%s,%s,%s,%s\n' "${label//,/_;}" "$status" "$c" "$l" "$e" "$nu" "$rc" >> "$TMP"

    case "$status" in
        OK|COMPILE_FAIL) ;;
        NUMERIC_FAIL) numeric_fail=$((numeric_fail + 1)) ;;
        NUMERIC_UNVERIFIED) numeric_unverified=$((numeric_unverified + 1)) ;;
        NUMERIC_LAYOUT_CANDIDATE) layout_candidate=$((layout_candidate + 1)) ;;
        *) infra_fail=$((infra_fail + 1)) ;;   # includes LOAD_FAIL, EVAL_FAIL, RUN_ERROR
    esac
done

mv "$TMP" "$CSV" || exit 1
chmod 644 "$CSV" 2>/dev/null || true
trap - EXIT

if [[ $LIST -eq 1 ]]; then
    echo "run_survey: $n cases -> $CSV (list mode; no ANE, no logs)"
    exit $((infra_fail > 0 ? 1 : 0))
fi

survey_rc=0
if (( numeric_fail > 0 || infra_fail > 0 )); then survey_rc=1; fi
if (( (numeric_unverified > 0 || layout_candidate > 0) && ALLOW_UNVERIFIED == 0 )); then survey_rc=1; fi

GIT_COMMIT=$(git -C "$PWD/.." rev-parse HEAD 2>/dev/null || echo unknown)
if [[ -n "$(git -C "$PWD/.." status --porcelain 2>/dev/null)" ]]; then
    GIT_DIRTY=1
else
    GIT_DIRTY=0
fi

{
    echo "timestamp=$STAMP"
    echo "mode=run"
    echo "probe=$(portable "$PROBE")"
    echo "cases=$n"
    echo "csv=$(portable "$CSV")"
    echo "errors=$(portable "$RAW")"
    echo "stdout=$(portable "$STDOUT_DIR")"
    echo "layouts=${LAYOUTS:-auto}"
    echo "dump=$DUMP"
    echo "allow_unverified=$ALLOW_UNVERIFIED"
    echo "git_commit=$GIT_COMMIT"
    echo "git_dirty=$GIT_DIRTY"
    echo "numeric_fail=$numeric_fail"
    echo "numeric_unverified=$numeric_unverified"
    echo "layout_candidate=$layout_candidate"
    echo "infra_fail=$infra_fail"
    echo "exit=$survey_rc"
} > "$RESULTS/run_meta.txt"

awk -F, 'NR>1 { c[$2]++ } END { for (k in c) printf "%s,%d\n", k, c[k] }' "$CSV" \
    | sort > "$RESULTS/summary.txt"
{
    echo "numeric_fail=$numeric_fail"
    echo "numeric_unverified=$numeric_unverified"
    echo "layout_candidate=$layout_candidate"
    echo "infra_fail=$infra_fail"
    echo "exit=$survey_rc"
} >> "$RESULTS/summary.txt"

echo "run_survey: $n cases -> $CSV"
echo "run_survey: raw errors -> $RAW"
echo "run_survey: stdout -> $STDOUT_DIR"
sed 's/^/  /' "$RESULTS/summary.txt"
if (( survey_rc != 0 )); then
    echo "run_survey: FAIL (numeric or runtime/infra regression; compile/load refusals are allowed)" >&2
fi
exit "$survey_rc"
