# Density-based Counterfactual Explanations for Graph Classification

Code for the paper "Counterfactual Explanations for Graph Classification Through the Lenses of Density", accepted for publication at the 2023 World Conference on eXplainable Artificial Intelligence (xAI 2023).
In this paper we turn our attention to some of the main characterizing features of real-world networks, such as the tendency to close triangles, the existence of recurring motifs, and the organization into dense modules. 
We thus define a general density-based counterfactual search framework to generate instance-level counterfactual explanations for graph classifiers, which can be instantiated with different notions of dense substructures. 
In particular, we show two specific instantiations of this framework: a method that searches for counterfactual graphs by opening or closing triangles, and a method driven by maximal cliques.

Use the two notebooks to run the different implementations of the framework (*CLI Example* notebook for the density-based methods, and *EDG+TRI+DAT Example* notebook for the baseline methods) on the 7 brain datasets provided:

1. *autism_graphs*: Autism Brain Image Data Exchange (ABIDE) project;
2. *bip80_graphs*: lithium response in type I bipolar disorder patients;
3. *adhd_90_graphs*: derived from the Multimodal Treatment of Attention Deficit Hyperactivity Disorder (MTA) project (ADHD vs Typically Developed);
4. *adhdm_90_graphs*: derived from the Multimodal Treatment of Attention Deficit Hyperactivity Disorder (MTA) project (Marijuana use vs Marijuana not used);
5. *OHSU_graphs*: Attention Deficit Hyperactivity Disorder classification;
6. *Peking_1_graphs*: Hyperactive Impulsive classification;
7. *KKI_graphs*: gender classification.
