

# olympus/ci_scripts/check_lint.sh 
#!/bin/bash  
# Runs pants lint on the codebase. 
set -e
pants --filter-address-regex='experimental/.*BUILD' lint ::
pants --filter-address-regex='-experimental/.*' lint ::

