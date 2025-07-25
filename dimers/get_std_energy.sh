#!/bin/bash

sed -n '/F12 singles correction/,/\*\*\*/p' $1
