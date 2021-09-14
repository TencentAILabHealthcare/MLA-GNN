### For each split of the features, perform WGCNA and generate adjacency matrix. 

data_folder = "./data/RNAseq_graph/RNAseq"
output_folder = "./data/RNAseq_graph/wgcna_output"

cancer = "GBMLGG"
data_file = paste(data_folder, cancer, "split15_train_320d_features_labels.csv", sep='/')
dir.create(paste(output_folder, cancer, sep='/'))
# WGCNA parameters
wgcna_power = 6
wgcna_minModuleSize = 10
wgcna_mergeCutHeight = 0.25
data = read.csv(data_file, header=F) # each row is a patient
geneExp = as.matrix(data[2:dim(data)[1], 83:322])

# gene as columns for WGCNA
# geneExp = t(geneExp)
dim(geneExp)

## imputate the NA by zero values.
geneExp[is.na(geneExp)]<-0

library(WGCNA)
adjacency = adjacency(geneExp, power = wgcna_power)
write.csv(adjacency,file=paste(output_folder, cancer, "split15_adjacency_matrix.csv", sep='/'),quote=F,row.names = F)

