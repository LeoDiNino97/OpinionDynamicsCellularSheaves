This repository aims at realizing a Python package to reproduce the models proposed in [Opinion dynamics over discourse sheaves](https://arxiv.org/pdf/2005.12798). 
In this paper Jacob Hansen and Robert Ghrist used graph cellular sheaf as a very expressive framework subsuming opinion dynamics via Laplacian heat diffusion equation in a new way.

Considering a graph $G(V,E)$, a cellular sheaf $\mathcal{F}$ on a graph is made up of
+ A vectorial space $\mathcal{F}_v$ for each node $v \in V$,
+ A vectorial space $\mathcal{F}_e$ for each edge $e \in E$,
+ A linear map $\mathcal{F}_{v \triangleleft e} : \mathcal{F}_v \rightarrow \mathcal{F}_e$ for each incidency $v \triangleleft e$, for each node $v \in V$, for each edge $e \in E$.
  
The block matrix collecting all the linear maps according to a fixed oriented incidency is called the coboundary map $\delta$: the sheaf laplacian can be derived from the coboundary map similarly to how we derive the graph laplacian from the incidency matrix: 

$$ L_{\mathcal{F}} = \delta^T \delta $$

The most obvious vectorial spaces to derive from this definition are the so called space of cochains: they result from a direct sum of the spaces (namely stalks) defined over the nodes and the edges respectively, so that an element belonging to a space of cochain is just the stack of the signals defined over all the nodes or the edges:

$$C^0(G,\mathcal{F}) = \bigoplus_{v \in V} \mathcal{F}_v$$

$$C^1(G,\mathcal{F}) = \bigoplus_{e \in E} \mathcal{F}_e$$

The idea is based on the following scheme:
+ Private-opinion expressions are modeled as 0-cochains: $x \in C^0(G,\mathcal{F})$: in particular, each node stalk is the space of each agents' private opinions;
+ Joint-opinion expressions over edges are modeled as 1-cochain: $\xi \in C^1(G,\mathcal{F})$: in particular, edge stalks are the spaces where pair-wise communications between agents lead to shared opinions;
+ Restriction maps are the way agents express their opinion from their private basis to the shared one.

In this way we can provide many interesting new flavour to classic opinion dynamics, modeling stubborn agents, external input, expression dynamic and joint dynamic of expression and opinion starting from the following ODEs system fixing a vector of initial private opinions $x_0$ and an initial coboundary map $\delta_0$:

$$
\begin{cases}
    \frac{dx}{dt} = -\alpha \delta^T\delta x \\
    \frac{d \delta}{dt} = -\beta \Pi_G(\delta xx^T)\\
    x(0) = x_0 \\
    \delta(0) = \delta_0
\end{cases}
$$




