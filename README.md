# godeep-l3-conde

The CONtrastive DEcomposer(ConDe) is a methodology which aims to identify some specific semantic information for a given input. In the geological context, our goal in using ConDe is to, given a seismic image, be able to encode it's structure and resolution/noise as separate, independent, latent vectors. By correclty finding these vectors, we gain a couple of possible benefits:

1. My swapping the resolution vecotr of different seismics A and B, we can create a new seismic with the structure of A and resolution of B;
2. By finding a good structure latent space, we can transform all seismics, regardless of it's acquisition methods, into the same distribution, thus making it easier for DL networks trained on one seismic to work on another.


ConDe is still under development. An explanation of the methodology, and some results and examples will be included here once the development is finished.
