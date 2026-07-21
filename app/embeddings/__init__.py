# Embedding generation module.
#
# Pipeline position:
#   KnowledgeDocument[]
#       -> EmbeddingService (via EmbeddingProvider)
#       -> EmbeddedDocument[]
#       -> (next epic: Qdrant indexing)
