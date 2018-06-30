
## First version

```
docker build -t solderzzc/rocketchat:tf_1.6_fast .
docker push solderzzc/rocketchat:tf_1.6_fast
```

```
Fri Jun 29 22:14:06 UTC 2018 : === Output wheel file is in: /root/tensorflow_temp
```

## Runtime 

```
docker build -f Dockerfile.aarch64 -t solderzzc/rocketchat:tf_1.6g_runtime .
docker push  solderzzc/rocketchat:tf_1.6g_runtime
```
