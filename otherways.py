from csc import divisi2
from divisi2.blending import blend
from csc.util.persist import PickleDict

pd = PickleDict('./pickledir/')

def NotOverRide(game, object):
    if pd.has_key(game):
        overlist = pd[game]['over']
    else:
        file = open(game + '.over', 'r')
        overlist = file.readlines()
        file.close()
        overlist = [x.lower().replace('\n', '') for x in overlist]
    if object in overlist: return False
    return True

def clear_top_items(sim, game, num):
    all = sim.top_items(num*3)
    return [x for x in all if NotOverRide(game, x)][:num]
    
def top_game_sims(game, object, n=6):
    if pd.has_key(game):
        similarity = pd[game]['blend']
    else:
        similarity = make_blend(game + '.pickle')
    frame = divisi2.SparseVector.from_counts([object])
    return clear_top_items(similarity.right_category(frame), game, n)

def make_blend(thefile):
    conceptnet = divisi2.network.conceptnet_matrix('en')
    thegame = divisi2.load(thefile).normalize_all()
    blended_matrix = blend(conceptnet, thegame)
    u,s,v = blended_matrix().svd()
    
    similarity = divisi2.reconstruct_similarity(u,s)
    pd[thefile.split('.')[0]] = similarity
    return similarity
