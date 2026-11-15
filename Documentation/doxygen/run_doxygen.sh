#!/bin/bash

# Upload the docs to my website
upload=false

# Run doxygen
doxygen doxyfile

# Upload to website
if [[ $upload == true ]]; then
  rsync -r html/* webpage:fml/doxygen/
fi
