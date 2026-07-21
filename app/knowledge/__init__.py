# Knowledge Engine — RAG ingestion pipeline.
#
# Pipeline order:
#   JsonKnowledgeLoader → KnowledgeValidator → KnowledgeNormalizer
#   → SectionExtractor → InMemoryKnowledgeRepository → KnowledgeSearchService
