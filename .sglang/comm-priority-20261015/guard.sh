# Sourced by the run scripts. Another session shares these GPUs: never start a measurement while a
# foreign process holds a GPU, and discard any run during which one appeared.
foreign() { nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -v prio_bench | grep -c . ; }
wait_idle() {
  local quiet=0 announced=0
  while true; do
    if [ "$(foreign)" = "0" ]; then quiet=$((quiet + 1)); [ $quiet -ge 3 ] && return; else
      quiet=0; [ $announced = 0 ] && echo "   (waiting: foreign GPU process present)" && announced=1; fi
    sleep 5
  done
}
# guarded <cmd...>: runs cmd while sampling for foreign GPU users; returns 99 if contaminated
guarded() {
  local flag; flag=$(mktemp)
  ( while kill -0 $$ 2>/dev/null; do [ "$(foreign)" != "0" ] && echo 1 > $flag; sleep 1; done ) &
  local sampler=$!
  "$@"; local rc=$?
  kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
  if [ -s $flag ]; then rm -f $flag; return 99; fi
  rm -f $flag; return $rc
}
