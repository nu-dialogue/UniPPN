#!/bin/bash

# Define the mapping
declare -A mapping=(
    ["nlu"]="u"
    ["dst"]="t"
    ["policy"]="p"
    ["nlg"]="g"
)

# Function to translate module names
modules2id() {
    local module_names="$1"
    local output=""
    for module_name in $module_names; do
        output+="${mapping[$module_name]}"
    done
    echo "$output"
}
