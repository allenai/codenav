# Elasticsearch

## Download

You can download Elasticsearch by running
```bash
python -m codenav.retrieval.elasticsearch.install_elasticsearch
```
This will save Elastic search to `codenav/external_src/elasticsearch-8.12.0`. 

## Start 

Once downloaded, you can start the Elasticsearch server by running:
```bash
bash codenav/external_src/elasticsearch-8.12.0/bin/elasticsearch
``` 

## Graphical interface

It can be useful to use Kibana, an GUI for Elasticsearch, so you can do things like
 deleting an index without running commands via Python. Kibana will automatically be downloaded when you run the above
 install script. You can start Kibana by running:
```bash
bash codenav/external_src/kibana-8.12.0/bin/kibana
```
and you can access the web interface by navigating to [http://localhost:5601](http://localhost:5601) in your browser.