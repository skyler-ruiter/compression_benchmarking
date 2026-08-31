# helper: resolve src dir arg + source common
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
. "${HERE}/_common.sh"
SRC="${1:?usage: $0 <source-dir>}"
cd "$SRC"
