from neo4j.v1 import GraphDatabase
import jsonlines, json, os

# query = '''
# MATCH p=(a:Gene {name: 'FGF6'})-[:INTERACTS_GiG*1..2]->(b:Gene)
# WITH *, relationships(p) as r
# RETURN r, a, b LIMIT 20
# '''

# query = '''
# MATCH (a:Gene {name:'FGF6'})-[r]-(b:Gene)
# RETURN TYPE(r), r, a, b, (startNode(r)=a) as out_n
# LIMIT 20
# '''

# query = '''
# MATCH (a:Gene {name: 'FGF6'})-[r]-(b)
# RETURN TYPE(r) AS rel, b.name AS name
# LIMIT 20
# '''

# MATCH (a:Gene {name:'FGF6'})-[r]-(b:Gene) RETURN TYPE(r), r, a, b, (startNode(r)=a) as out_n

#MATCH (a:Gene {name:'FGF6'})-[r]-(b:Gene) RETURN TYPE(r), r, a, b

#MATCH (a:Gene {name:'FGF6'})-[r]-(b:Gene) RETURN TYPE(r), a, b

#MATCH (a:Gene {name:'FGF6'})-[r]-(b:Gene)
# RETURN r, a, 

#MATCH (n:Foo)-[r]-(m) WHERE n.id = "bar"
# RETURN n,m,type(r), (startNode(r) = n) as out_n

def build_query(e1, e2, limit):
    # query = '''
    #     MATCH path = allShortestPaths((a:Gene)-[r*1..4]-(b))
    #     WHERE a.name =~ '(?i)%s.*' AND b.name =~ '(?i)%s.*' AND (LABELS(b)=["Disease"] OR LABELS(b)=["SideEffect"] OR LABELS(b)=["Symptom"])
    #     RETURN [n in nodes(path) | n.name] AS stops, 
    #     [x in relationships(path) |TYPE(x)] AS reltypes, 
    #     [n in nodes(path) | LABELS(n)] as nodelabels, 
    #     length(path) AS len
    #     LIMIT %s
    #     ''' % (e1, e2, limit)
    # query = '''
    #     MATCH path = allShortestPaths((a:Gene)-[r*1..4]-(b:Gene))
    #     WHERE a.name =~ '(?i)%s.*' AND b.name =~ '(?i)%s.*'
    #     RETURN [n in nodes(path) | n.name] AS stops, 
    #     [x in relationships(path) |TYPE(x)] AS reltypes, 
    #     [n in nodes(path) | LABELS(n)] as nodelabels, 
    #     length(path) AS len
    #     LIMIT %s
    #     ''' % (e1, e2, limit)
    query = '''
        MATCH path = allShortestPaths((a:Compound)-[r*1..4]-(b:Compound))
        WHERE a.name =~ '(?i)%s.*' AND b.name =~ '(?i)%s.*'
        RETURN [n in nodes(path) | n.name] AS stops, 
        [x in relationships(path) |TYPE(x)] AS reltypes, 
        [n in nodes(path) | LABELS(n)] as nodelabels, 
        length(path) AS len
        LIMIT %s
        ''' % (e1, e2, limit)
    return query

def build_query_ex(e1, e2, limit):
    query = '''
        MATCH path = allShortestPaths((a)-[r*1..4]-(b))
        WHERE a.name =~ '(?i)%s.*' AND b.name =~ '(?i)%s.*'
        RETURN [n in nodes(path) | n.name] AS stops, 
        [x in relationships(path) |TYPE(x)] AS reltypes, 
        [n in nodes(path) | LABELS(n)] as nodelabels, 
        length(path) AS len
        LIMIT %s
        ''' % (e1, e2, limit)
    return query

def run_query(pairs):
    driver = GraphDatabase.driver("bolt://neo4j.het.io")
    with driver.session() as session:
        for pair in pairs:
            pairdic={}
            pairdic['orig_tuple']=pair[0]
            pairdic['node_match']=pair[1]
            e1=pair[1][0]
            e2=pair[1][1]
            #e1, e2 = 'TGFB1', 'breast cancer'
            pname=e1+'##'+e2
            print ('\n', pname, os.path.isfile(pname+'.json'))
            if 'xnoma#tchx' in e1 or 'xnoma#tchx' in e2:
                pass
            else:
                if not os.path.isfile(pname+'.json'):
                    try:
                        query = build_query(e1.lower(), e2.lower(), '50')
                        print (query)
                        result = session.run(query)
                        paths= []
                        for idp, r in enumerate(result):
                            dic={}
                            dic["pathid"]=idp
                            dic["stops"] = r["stops"]
                            dic["reltypes"] = r["reltypes"]
                            dic["nodelabels"] = [x for xs in r["nodelabels"] for x in xs]
                            dic["len"] = r["len"]
                            paths.append(dic)
                            # print (dic)
                        pairdic['paths']=paths

                        if len(paths) < 1:
                            print (f'Result: {len(paths)}, run ex query')
                            query = build_query_ex(e1.lower(), e2.lower(), '50')
                            print (query)
                            result = session.run(query)
                            paths= []
                            for idp, r in enumerate(result):
                                dic={}
                                dic["pathid"]=idp
                                dic["stops"] = r["stops"]
                                dic["reltypes"] = r["reltypes"]
                                dic["nodelabels"] = [x for xs in r["nodelabels"] for x in xs]
                                dic["len"] = r["len"]
                                paths.append(dic)
                                # print (dic)
                            pairdic['paths']=paths

                        with open(pname+'.json', 'w') as ff:
                            json.dump(pairdic, ff)
                        print (f"#Saved to {pname}")
                    except:
                        pass

def load_jsonl(fname):
    pairs=[]
    with jsonlines.open(fname, 'r') as fop:
        for en, row in enumerate(fop):
            # print (en, row['orig_tuple'], row['node_match'] )
            pairs.append([row['orig_tuple'], row['node_match']])
    return pairs

# pairs = load_jsonl('comagc_nodes.jsonl')
pairs = load_jsonl('drug_nodes.jsonl')
# print (pairs[:3][0])
run_query(pairs)
# l = [['Gene'], ['Gene'], ['Disease']]
# print ([x for xs in l for x in xs])