#!/bin/bash

set -euo pipefail

cat <<'EOF'
ERROR: trex/scripts/run_tsmc_baseline.sh is archived.

The project pivoted in March 2026 away from twisted/transport-era TSMC + value-head baselines.
This runner is intentionally disabled to prevent accidental use.

For historical context, see: docs/archive/2026-feb/
EOF

exit 1
