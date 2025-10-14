
| Service                                  | Function                                                         | Example Tech                                              |
| ---------------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------- |
| **User Service**                         | User registration, profiles, relationships (follow/friend graph) | PostgreSQL + Redis                                        |
| **Post Service**                         | Create / read / update / delete posts (images, text, videos)     | Cassandra / DynamoDB                                      |
| **Feed Service**                         | Aggregates posts from followed users                             | Kafka + Redis + GraphQL                                   |
| **Like & Comment Service**               | Handles engagement data                                          | MongoDB / DynamoDB                                        |
| **Notification Service**                 | Push/email/real-time alerts                                      | Firebase / SNS / Kafka                                    |
| **Search Service**                       | User and content search                                          | Elasticsearch / OpenSearch                                |
| **Media Service**                        | Stores and delivers images/videos                                | S3 + CloudFront                                           |
| **Recommendation Service (LLM-powered)** | Personalized feed and content suggestions                        | PyTorch + LangChain + Vector DB (e.g., Pinecone / Milvus) |