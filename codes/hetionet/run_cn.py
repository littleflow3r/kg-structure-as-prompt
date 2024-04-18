from neo4j.v1 import GraphDatabase
import jsonlines, json, os

# MATCH (a:Gene)-[r1]-(b)-[r2]-(c)
# WHERE a.name =~ '(?i)erbb2.*' AND c.name =~ '(?i)breast cancer.*'
# AND (LABELS(c)=["Disease"] OR LABELS(c)=["SideEffect"] OR LABELS(c)=["Symptom"])
# RETURN TYPE(r1) as brela, b.name as bname, LABELS(b) as blabel, TYPE(r2) as brelc

# MATCH (a)-[r1]-(b)-[r2]-(c)
# WHERE a.name =~ '(?i)erbb2.*' AND c.name =~ '(?i)breast cancer.*'
# RETURN TYPE(r1) as brela, b.name as bname, LABELS(b) as blabel, TYPE(r2) as brelc

def build_query(e1, e2, limit):
    # query = '''
    #     MATCH (a:Gene)-[r1]-(b)-[r2]-(c)
    #     WHERE a.name =~ '(?i)%s.*' AND c.name =~ '(?i)%s.*' 
    #     AND (LABELS(c)=["Disease"] OR LABELS(c)=["SideEffect"] OR LABELS(c)=["Symptom"])
    #     RETURN TYPE(r1) as brela, b.name as bname, LABELS(b) as blabel, TYPE(r2) as brelc
    #     LIMIT %s
    #     ''' % (e1, e2, limit)
    # query = '''
    #     MATCH (a:Gene)-[r1]-(b)-[r2]-(c:Gene)
    #     WHERE a.name =~ '(?i)%s.*' AND c.name =~ '(?i)%s.*' 
    #     RETURN TYPE(r1) as brela, b.name as bname, LABELS(b) as blabel, TYPE(r2) as brelc
    #     LIMIT %s
    #     ''' % (e1, e2, limit)
    query = '''
        MATCH (a:Compound)-[r1]-(b)-[r2]-(c:Compound)
        WHERE a.name =~ '(?i)%s.*' AND c.name =~ '(?i)%s.*' 
        RETURN TYPE(r1) as brela, b.name as bname, LABELS(b) as blabel, TYPE(r2) as brelc
        LIMIT %s
        ''' % (e1, e2, limit)
    return query

def build_query_ex(e1, e2, limit):
    query = '''
        MATCH (a)-[r1]-(b)-[r2]-(c)
        WHERE a.name =~ '(?i)%s.*' AND c.name =~ '(?i)%s.*' 
        RETURN TYPE(r1) as brela, b.name as bname, LABELS(b) as blabel, TYPE(r2) as brelc
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
                        query = build_query(e1.lower(), e2.lower(), '100')
                        print (query)
                        result = session.run(query)
                        paths= []
                        for idp, r in enumerate(result):
                            dic={}
                            dic["cnid"]=idp
                            dic["cn_name"] = r["bname"]
                            dic["cn_label"] = r["blabel"]
                            dic["brela"] = r["brela"]
                            dic["brelc"] = r["brelc"]
                            paths.append(dic)
                            # print (dic)
                        pairdic['paths']=paths

                        if len(paths) < 1:
                            print (f'Result: {len(paths)}, run ex query')
                            query = build_query_ex(e1.lower(), e2.lower(), '100')
                            print (query)
                            result = session.run(query)
                            paths= []
                            for idp, r in enumerate(result):
                                dic={}
                                dic["cnid"]=idp
                                dic["cn_name"] = r["bname"]
                                dic["cn_label"] = r["blabel"]
                                dic["brela"] = r["brela"]
                                dic["brelc"] = r["brelc"]
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
# pairs = load_jsonl('gene_nodes2.jsonl')
pairs = load_jsonl('drug_nodes.jsonl')
# print (pairs[:3][0])
run_query(pairs)
# l = [['Gene'], ['Gene'], ['Disease']]
# print ([x for xs in l for x in xs])