for target in target/debug/deps/valgrind_* ; do
  valgrind --track-origins=yes $@ $target
done

