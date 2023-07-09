class GPTBench:
    def __init__(self, bench_name="charLM-default"):
        # load settings and search space
        # prepare data
        pass

    def get_search_space(self):
        # retrieve the search space as per the bench_name
        pass

    def query(self, /):
        # inputs a config
        # update settings
        # instantiat model, optimizer, scheduler --- load checkpoint here, if applicable
        # run 1 function evaluation
        # return loss (minimze) and cost
        return None
    
    def convert_search_space(self, target: str="neps"):
        ss = self.get_search_space()
        if target == "neps":
            import neps
            # convert to NePS pipeline_space
            pass 
        return ss
