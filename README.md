# CrossE

this is the code of CrossE in paper "Interaction Embeddings for Prediction and Explanation in Knowledge Graphs"(WSDM 2019)

## INTRODUCTION

paper: Interaction Embeddings for Prediction and Explanation in Knowledge Graphs. (WSDM'2019)

## RUN

example: python3 CrossE.py --batch 4000 --data ../data/FB15k/ --dim 300 --eval_per 20 --loss_weight 1e-6 --lr 0.01 --max_iter 500 --save_per 20 

## DATASET

There are three benchmark datasets used in this paper, WN18, FB15k and FB15k-237. They are compressed in data.zip.

## CITE

If the codes help you or the paper inspire your, please cite following paper:

Wen Zhang, Bibek Paudel, Wei Zhang, Abraham Bernstein and Huajun Chen. Interaction Embeddings for Prediction and Explanation in Knowledge Graphs. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining (WSDM2019).

