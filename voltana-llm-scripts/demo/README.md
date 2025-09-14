# Simple Demo of VoltanaLLM

This is a simple demo to serve `Llama-3.1-8B-Instruct` with 2P2D configuration by VoltanaLLM on 4xA100 80G SXM4.

Start P/D instances (w/ EcoFreq):

```shell
# 0 1 2 3 are GPU index
./p_server.sh 8100 0 8998
./p_server.sh 8101 1 8999
./d_server.sh 8200 2
./d_server.sh 8201 3
```

Start the EcoRoute:

```shell
./router.sh config-2p2d.sh
```

Then run a simple test:

```shell
./client.sh config-2p2d.sh
```
