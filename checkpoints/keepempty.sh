#!/bin/sh
while true; do
  find * -type f -name "*.t7" |
  while read fileName; do
    netName=$(echo $fileName | sed -E "s/_[0-9]+_net_[D,G].t7//g"); # get network name
    find . -type f -name "$netName\_[0-9]*" | # find other checkpoints with that name
      awk -F_ '{ print $(NF-2), $0 }' |       # prepend the epoch number
      sort -rn -k1 -g |                       # sort
      cut -d" " -f 2 |                        # remove prepended number
      tail +$((10+1)) |                       # pipe anything more than 5 checkpoints old
      xargs rm                                # delete
  done
  sleep 180
done
