from elasticsearch import Elasticsearch
from pprint import pprint
import json

# 连接到Elasticsearch
# 对于ES 8.x with security disabled:
es = Elasticsearch(
    "http://localhost:9200",
    request_timeout=30
)

# 检查连接
if es.ping():
    print("成功连接到Elasticsearch")
else:
    print("无法连接到Elasticsearch")
    exit(1)
# 创建索引（类似于SQL中的表）
index_name = "users"

# 如果索引已存在则删除（仅用于实验）
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# 创建带有映射的索引
mapping = {
    "mappings": {
        "properties": {
            "name": {"type": "text"},
            "email": {"type": "keyword"},
            "age": {"type": "integer"},
            "interests": {"type": "text"},
            "created_at": {"type": "date"}
        }
    }
}

es.indices.create(index=index_name, body=mapping)

# 插入文档
users = [
    {"name": "Alice Smith", "email": "alice@example.com", "age": 30, "interests": ["reading", "hiking", "programming"]},
    {"name": "Bob Johnson", "email": "bob@example.com", "age": 25, "interests": ["gaming", "music"]},
    {"name": "Charlie Brown", "email": "charlie@example.com", "age": 35, "interests": ["sports", "cooking", "traveling"]},
    {"name": "Diana Prince", "email": "diana@example.com", "age": 28, "interests": ["yoga", "reading", "photography"]}
]

for i, user in enumerate(users):
    es.index(index=index_name, id=i+1, document=user)

# 刷新索引使文档可搜索
es.indices.refresh(index=index_name)

print("\n=== Elasticsearch 检索 ===")

# 1. 检索所有文档
print("\n1. 所有用户:")
result = es.search(index=index_name, query={"match_all": {}})
for hit in result['hits']['hits']:
    print(f"ID: {hit['_id']}, Name: {hit['_source']['name']}, Score: {hit['_score']}")

# 2. 全文搜索
print("\n2. 搜索 'reading' 兴趣的用户:")
result = es.search(index=index_name, query={"match": {"interests": "reading"}})
for hit in result['hits']['hits']:
    print(f"{hit['_source']['name']} - {hit['_source']['interests']}")

# 3. 布尔查询
print("\n3. 年龄大于28且喜欢阅读的用户:")
query = {
    "bool": {
        "must": [
            {"range": {"age": {"gt": 28}}},
            {"match": {"interests": "reading"}}
        ]
    }
}
result = es.search(index=index_name, query=query)
for hit in result['hits']['hits']:
    print(f"{hit['_source']['name']} (Age: {hit['_source']['age']})")

# 4. 聚合分析
print("\n4. 按年龄分组统计:")
aggs = {
    "age_groups": {
        "range": {
            "field": "age",
            "ranges": [
                {"to": 25},
                {"from": 25, "to": 30},
                {"from": 30}
            ]
        }
    }
}
result = es.search(index=index_name, aggs=aggs, size=0)
for bucket in result['aggregations']['age_groups']['buckets']:
    print(f"Age range: {bucket.get('key', 'N/A')}, Count: {bucket['doc_count']}")
