#!/bin/bash
set -e


sed -n '/F12 singles correction/,/\*\*\*/p' $1

tail -n 1 $1
