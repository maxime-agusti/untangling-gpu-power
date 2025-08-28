---
name: debian12-big
version: 2024090216
arch: x86_64
description: debian 12
author: jacquet.pierre
visibility: private
destructive: false
os: linux
image:
  file: /home/pjacquet/images/debian12-nvidia570.tar.zst
  kind: tar
  compression: zstd
postinstalls:
- archive: server:///grid5000/postinstalls/g5k-postinstall.tgz
  compression: gzip
  script: g5k-postinstall --net debian --disk-aliases --fstab nfs --restrict-user current
boot:
  kernel: "/vmlinuz"
  initrd: "/initrd.img"
  kernel_params: ''
filesystem: ext4
partition_type: 131
multipart: false