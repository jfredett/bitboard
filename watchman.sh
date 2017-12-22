watchman-make -p 'tests/*' --make 'clear; cargo' -t test \
              -p 'src/*.rs' 'src/**/*.rs' --make 'clear; cargo' -t test \
              -p 'Cargo.toml' --make 'clear; cargo' -t update
