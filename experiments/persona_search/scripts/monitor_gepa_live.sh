#!/bin/bash
# GEPA Optimization Live Monitor
# Shows real-time progress of multiple GEPA runs

REPO_ROOT="${TAU2BENCH_DATA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"

# ANSI colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m' # No Color

# Configuration for the 3 runs
declare -A RUNS=(
  ["p0003_v2"]="11084:data/persona_opt/gepa_runs/p0003_aggressive_20260606_145857:35:Distracted"
  ["p0001_v2"]="18162:data/persona_opt/gepa_runs/p0001_aggressive50_20260606_151059:50:Suspicious"
  ["p0002_v2"]="18216:data/persona_opt/gepa_runs/p0002_aggressive50_20260606_151059:50:Trusting"
)

# Draw progress bar
draw_bar() {
  local current=$1
  local total=$2
  local width=40
  local filled=$((current * width / total))
  local empty=$((width - filled))
  local percent=$((current * 100 / total))

  printf "["
  printf "${GREEN}%${filled}s${NC}" | tr ' ' '█'
  printf "%${empty}s" | tr ' ' '░'
  printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# Draw mini sparkline for score history
draw_sparkline() {
  local -a scores=("$@")
  local bars="▁▂▃▄▅▆▇█"

  if [ ${#scores[@]} -eq 0 ]; then
    echo "─"
    return
  fi

  # Find min/max
  local min=${scores[0]}
  local max=${scores[0]}
  for score in "${scores[@]}"; do
    (( $(echo "$score < $min" | bc -l) )) && min=$score
    (( $(echo "$score > $max" | bc -l) )) && max=$score
  done

  # Normalize and draw
  for score in "${scores[@]}"; do
    if (( $(echo "$max == $min" | bc -l) )); then
      printf "${bars:3:1}"
    else
      local norm=$(echo "($score - $min) / ($max - $min) * 7" | bc -l)
      local idx=$(printf "%.0f" "$norm")
      printf "${bars:$idx:1}"
    fi
  done
}

# Extract recent iterations from log
get_iteration_history() {
  local logfile=$1
  local count=${2:-10}

  if [ ! -f "$logfile" ]; then
    echo ""
    return
  fi

  # Get last N iterations with their scores
  grep "Best score on valset:" "$logfile" 2>/dev/null | tail -$count | \
    awk '{print $2, $NF}' | sed 's/://'
}

# Get current activity
get_current_activity() {
  local logfile=$1

  if [ ! -f "$logfile" ]; then
    echo "initializing..."
    return
  fi

  # Check last few lines for what's happening
  local last_lines=$(tail -20 "$logfile" 2>/dev/null)

  if echo "$last_lines" | grep -q "Continue to full eval"; then
    echo "running full valset eval"
  elif echo "$last_lines" | grep -q "New subsample score"; then
    echo "evaluating minibatch"
  elif echo "$last_lines" | grep -q "Selected program"; then
    echo "selecting next parent"
  elif echo "$last_lines" | grep -q "Valset score"; then
    echo "computing scores"
  else
    echo "iterating..."
  fi
}

# Get candidate pool summary
get_candidate_summary() {
  local rundir=$1
  local candidates_file="$rundir/candidates.json"

  if [ ! -f "$candidates_file" ]; then
    echo "0 candidates"
    return
  fi

  local count=$(python3 -c "import json; print(len(json.load(open('$candidates_file'))))" 2>/dev/null)
  echo "${count:-0} candidates"
}

# Main display function
display_dashboard() {
  clear

  # Header
  echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}║${NC}  ${BOLD}GEPA Persona Optimization - Live Monitor${NC}                              ${BOLD}${CYAN}║${NC}"
  echo -e "${BOLD}${CYAN}║${NC}  $(date '+%Y-%m-%d %H:%M:%S')                                                      ${BOLD}${CYAN}║${NC}"
  echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
  echo ""

  # Process each run
  for run_name in p0003_v2 p0001_v2 p0002_v2; do
    local config=${RUNS[$run_name]}
    local pid=$(echo $config | cut -d: -f1)
    local rundir=$(echo $config | cut -d: -f2)
    local taskcount=$(echo $config | cut -d: -f3)
    local seed_type=$(echo $config | cut -d: -f4)
    local logfile="$rundir/run_log.txt"

    echo -e "${BOLD}━━━ $run_name ${DIM}(Seed: $seed_type)${NC} ${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Check if process is alive
    if ! ps -p $pid > /dev/null 2>&1; then
      echo -e "  ${RED}✗ STOPPED${NC} (PID $pid not found)"
      echo ""
      continue
    fi

    local etime=$(ps -p $pid -o etime= | tr -d ' ')
    echo -e "  ${GREEN}✓ RUNNING${NC}  Runtime: ${CYAN}$etime${NC}  PID: $pid"

    # Parse progress from log
    if [ -f "$logfile" ]; then
      local last_iter=$(grep "Iteration [0-9]*:" "$logfile" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d ':')
      local full_evals=$(grep "Continue to full eval" "$logfile" 2>/dev/null | wc -l | tr -d ' ')
      local best_score=$(grep "Best score on valset:" "$logfile" 2>/dev/null | tail -1 | awk '{print $NF}')
      local seed_score=$(grep "Iteration 0:" "$logfile" -A 2 2>/dev/null | grep "Valset score" | awk '{print $NF}' | head -1)

      # Calculate metrics
      local seed_cost=$((taskcount * 2))
      local minibatch_cost=$((last_iter * 10))
      local fulleval_cost=$((full_evals * taskcount * 2))
      local total_calls=$((seed_cost + minibatch_cost + fulleval_cost))
      local remaining=$((1000 - total_calls))

      # Budget bar
      echo ""
      echo -e "  ${BOLD}Budget:${NC} $(draw_bar $total_calls 1000)"
      echo -e "          Remaining: ${YELLOW}$remaining${NC} calls"

      # Progress stats
      echo ""
      echo -e "  ${BOLD}Progress:${NC}"
      echo -e "    Iterations: ${CYAN}$last_iter${NC}  │  Full evals: ${CYAN}$full_evals${NC}  │  $(get_candidate_summary $rundir)"

      # Score evolution
      if [ -n "$best_score" ]; then
        local difficulty=$(echo "$best_score * 100" | bc -l | xargs printf "%.1f")
        local seed_diff=$(echo "${seed_score:-0} * 100" | bc -l | xargs printf "%.1f")
        local delta=$(echo "$difficulty - $seed_diff" | bc -l | xargs printf "%.1f")

        echo ""
        echo -e "  ${BOLD}Performance:${NC}"
        echo -e "    Best difficulty: ${GREEN}${difficulty}%${NC}  (seed: ${DIM}${seed_diff}%${NC}, Δ: ${YELLOW}${delta:+"+"}${delta}pp${NC})"

        # Get iteration history and draw sparkline
        local hist=$(get_iteration_history "$logfile" 15)
        if [ -n "$hist" ]; then
          local scores=($(echo "$hist" | awk '{print $2}' | xargs))
          local iters=($(echo "$hist" | awk '{print $1}' | xargs))

          echo -e "    Trend (last 15): $(draw_sparkline "${scores[@]}")"
          echo -e "                     ${DIM}iter ${iters[0]} → ${iters[-1]}${NC}"
        fi
      fi

      # Current activity
      local activity=$(get_current_activity "$logfile")
      echo ""
      echo -e "  ${BOLD}Status:${NC} ${BLUE}$activity${NC}"

      # Latest persona insight (if available)
      if [ -f "$rundir/candidates.json" ] && [ "$full_evals" -gt 0 ]; then
        local latest_persona=$(python3 << PYEOF 2>/dev/null
import json
try:
    cands = json.load(open('$rundir/candidates.json'))
    if len(cands) > 1:
        # Get latest accepted (last one that's not seed)
        latest = cands[-1]['persona_text']
        # Extract first new behavior added
        lines = [l.strip() for l in latest.split('\n') if l.strip().startswith('- ')]
        if len(lines) > 5:  # More than seed
            print(lines[5][:80] + "..." if len(lines[5]) > 80 else lines[5])
except:
    pass
PYEOF
)
        if [ -n "$latest_persona" ]; then
          echo -e "  ${DIM}Latest mutation:${NC}"
          echo -e "    ${DIM}$latest_persona${NC}"
        fi
      fi

    else
      echo -e "  ${YELLOW}⚠ Initializing${NC} (no log file yet)"
    fi

    echo ""
  done

  # System summary
  echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"
  echo -e "${BOLD}System Status:${NC}"

  local gepa_count=$(pgrep -f 'gepa_runner.py' | wc -l | tr -d ' ')
  local tau2_count=$(pgrep -f 'tau2 run' | wc -l | tr -d ' ')
  local claude_count=$(pgrep -f 'claude --print' | wc -l | tr -d ' ')

  echo -e "  GEPA runners: ${CYAN}$gepa_count${NC}  │  tau2: ${CYAN}$tau2_count${NC}  │  claude workers: ${CYAN}$claude_count${NC}"

  local load=$(uptime | awk '{print $(NF-2), $(NF-1), $NF}')
  echo -e "  System load: ${DIM}$load${NC}"

  # Check for errors
  local error_count=0
  for run_name in p0003_v2 p0001_v2 p0002_v2; do
    local config=${RUNS[$run_name]}
    local rundir=$(echo $config | cut -d: -f2)
    local logfile="$rundir/run_log.txt"

    if [ -f "$logfile" ]; then
      local errors=$(tail -100 "$logfile" 2>/dev/null | grep -i "error\|exception\|failed" | grep -v "LiteLLM" | wc -l | tr -d ' ')
      error_count=$((error_count + errors))
    fi
  done

  if [ $error_count -gt 0 ]; then
    echo -e "  ${YELLOW}⚠${NC} Recent errors: $error_count (check logs)"
  else
    echo -e "  ${GREEN}✓${NC} No recent errors"
  fi

  # Completion estimate
  echo ""
  echo -e "${BOLD}Estimated completion:${NC}"

  # Find the run with most remaining budget
  local max_remaining=0
  for run_name in p0003_v2 p0001_v2 p0002_v2; do
    local config=${RUNS[$run_name]}
    local rundir=$(echo $config | cut -d: -f2)
    local taskcount=$(echo $config | cut -d: -f3)
    local logfile="$rundir/run_log.txt"

    if [ -f "$logfile" ]; then
      local last_iter=$(grep "Iteration [0-9]*:" "$logfile" 2>/dev/null | tail -1 | awk '{print $2}' | tr -d ':')
      local full_evals=$(grep "Continue to full eval" "$logfile" 2>/dev/null | wc -l | tr -d ' ')
      local total_calls=$(( taskcount * 2 + last_iter * 10 + full_evals * taskcount * 2 ))
      local remaining=$((1000 - total_calls))

      if [ $remaining -gt $max_remaining ]; then
        max_remaining=$remaining
      fi
    fi
  done

  if [ $max_remaining -gt 500 ]; then
    echo -e "  ${YELLOW}~6-8 hours${NC} (early stage)"
  elif [ $max_remaining -gt 200 ]; then
    echo -e "  ${YELLOW}~2-4 hours${NC}"
  elif [ $max_remaining -gt 0 ]; then
    echo -e "  ${CYAN}~1-2 hours${NC}"
  else
    echo -e "  ${GREEN}All runs complete!${NC}"
  fi

  echo ""
  echo -e "${DIM}Press Ctrl+C to exit  │  Refresh: 30s${NC}"
}

# Main loop
echo "Starting GEPA Live Monitor..."
echo "Monitoring 3 runs: p0003_v2, p0001_v2, p0002_v2"
echo ""

while true; do
  display_dashboard
  sleep 30
done
