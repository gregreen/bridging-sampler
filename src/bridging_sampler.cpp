
#include "bridging_sampler.h"



bridgesamp::BridgingSampler::BridgingSampler(
        uint16_t n_dim,
        uint16_t n_samples,
        std::function<double(const std::vector<uint16_t>&)> logp_node)
    : n_dim(n_dim), n_samples(n_samples),
      eval_node(logp_node),
      r_dim(0,n_dim-1), r_samp(0,n_samples-1),
      r_uniform(0., 1.)
{
    // Seed pseudo-random number generator
    std::random_device rd;
    r.seed(rd());
    
    // Allocate memory to sample workspace
    _samp_ws.resize(n_samples);
    _samp_ws_dbl.resize(n_samples);
    _dim_ws.resize(n_dim);

    // Transition probabilities
    b_prob.reserve(n_dim);
    f_prob.reserve(n_dim);

    b_prob.push_back(0.5);
    f_prob.push_back(0.);
    for(int i=0; i<n_dim-1; i++) {
        b_prob.push_back(0.4);
        f_prob.push_back(0.6);
    }
    b_prob.push_back(0.);
    f_prob.push_back(1.);
    
    // Miscellaneous quantities of use
    p0 = 1.;
    logp0 = 0.;
    log_n_samples = std::log((double)n_samples);
    state_rank = 0;
}


void bridgesamp::BridgingSampler::step() {
    std::cout << "Entering step()" << std::endl;

    // Choose a step type
    double x = r_uniform(r);
    double b = b_prob.at(state_rank);
    double f = f_prob.at(state_rank);
    
    std::cout << "State rank: " << state_rank << std::endl;
    std::cout << " - b: " << b << std::endl
              << " - f: " << f << std::endl;
    

    if(x < b) {
        std::cout << "Chose to transition backward" << std::endl;
        transition_backward();
    } else if(x < f+b) {
        std::cout << "Chose to transition forward" << std::endl;
        transition_forward();
    } else {
        std::cout << "Chose to take a Gibbs step" << std::endl;
        lazy_gibbs_choose_dim();
    }
    
    std::cout << "Exiting step()" << std::endl;
}


uint16_t bridgesamp::BridgingSampler::get_n_dim() const {
    return n_dim;
}


uint16_t bridgesamp::BridgingSampler::get_n_samples() const {
    return n_samples;
}


const std::vector<uint16_t>& bridgesamp::BridgingSampler::get_state() const {
    return state->first;
}


void bridgesamp::BridgingSampler::randomize_state() {
    std::cout << "Entering randomize_state()" << std::endl;

    // Choose random sample for each dimension
    for(auto& s : _dim_ws) {
        s = r_samp(r);
    }
    
    // Find node
    auto it = node.find(_dim_ws);

    // Create node if it doesn't exist
    if(it == node.end()) {
        std::cout << "Creating node";
        for(auto s : _dim_ws) {
            std::cout << " " << s;
        }
        std::cout << std::endl;

        double logp = eval_node(_dim_ws);
        it = node.insert(
            std::make_pair(_dim_ws, bridgesamp::Node{std::exp(logp), logp})
        ).first;

        percolate_up(it);
    }

    // Update the state to point to this node
    state = it;
    state_rank = 0; // At 0th level in hierarchy
    
    std::cout << "Exiting randomize_state()" << std::endl;
}


void bridgesamp::BridgingSampler::_lazy_gibbs_inner(
        std::vector<uint16_t>& starting_state,
        uint16_t dim)
{
    std::cout << "Entering _lazy_gibbs_inner()" << std::endl;

    std::cout << "Starting state:";
    for(auto s : starting_state) {
        std::cout << " " << s;
    }
    std::cout << std::endl;

    // Find ln(p) of states that have already been explored
    _samp_ws.clear();     // Will hold explored sample numbers
    _samp_ws_dbl.clear(); // Will hold corresponding ln(p)

    for(uint16_t s=0; s<n_samples; s++) {
        starting_state[dim] = s;

        auto it = node.find(starting_state);
        if(it != node.end()) {
            _samp_ws.push_back(s);
            _samp_ws_dbl.push_back(it->second.logp);
        }
    }

    uint16_t n_unexplored = n_samples - _samp_ws.size();

    std::cout << n_unexplored << " of " << n_samples << " states unexplored." << std::endl;
    
    // Get maximum ln(p) of explored states
    double lnp_max = *(std::max_element(
        _samp_ws_dbl.begin(),
        _samp_ws_dbl.end())
    );
    
    // Find total ln(p) of explored states
    double lnp_explored = 0.;

    for(auto lnp : _samp_ws_dbl) {
        lnp_explored += std::exp(lnp - lnp_max);
    }

    lnp_explored = std::log(lnp_explored) + lnp_max;

    // Calculate total ln(p) of unexplored states
    double ln_n_unexplored = std::log((double)(n_samples-_samp_ws.size()));
    double lnp_unexplored = get_lnp0_of_rank(state_rank) + ln_n_unexplored;

    double lnp_tot = add_logs(lnp_unexplored, lnp_explored);

    std::cout << "p(explored) = "
              << std::exp(lnp_explored-lnp_tot) << std::endl;

    // Decide whether to pick explored state
    if(!n_unexplored || (std::log(r_uniform(r)) < lnp_explored-lnp_tot)) {
        std::cout << "Picking explored state." << std::endl;

        // Pick explored state
        for(auto& lnp : _samp_ws_dbl) {
            lnp = std::exp(lnp - lnp_max);
        }
        std::discrete_distribution<uint16_t> d(_samp_ws_dbl.begin(), _samp_ws_dbl.end());

        starting_state[dim] = _samp_ws[d(r)];
        state = node.find(starting_state);
        
        std::cout << "Picked explored state:";
        for(auto s : starting_state) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
    } else {
        std::cout << "Picking unexplored state." << std::endl;

        // Pick unexplored state at random
        std::uniform_int_distribution<uint16_t> d(0, n_unexplored-1);
        uint16_t i_pick = d(r);

        std::cout << "Scanning for unexplored state #"
                  << i_pick << ":" << std::endl;

        // Scan through to find i_pick'th unexplored sample
        int32_t idx = 0;
        int32_t i_unexplored = -1;
        auto it_explored = _samp_ws.begin();
        for(; i_unexplored != i_pick; idx++) {
            std::cout << "- sample " << idx;
            if((it_explored != _samp_ws.end()) && (*it_explored == idx)) {
                std::cout << ": explored" << std::endl;
                ++it_explored;
            } else {
                std::cout << ": unexplored" << std::endl;
                ++i_unexplored;
            }
        }

        // Add the new state
        starting_state[dim] = idx-1;
        state = get_node(starting_state);
        
        std::cout << "Picked unexplored state:";
        for(auto s : starting_state) {
            std::cout << " " << s;
        }
        std::cout << std::endl;
    }

    std::cout << "Exiting _lazy_gibbs_inner()" << std::endl;
}


void bridgesamp::BridgingSampler::lazy_gibbs(uint16_t dim) {
    std::cout << "Entering lazy_gibbs()" << std::endl;

    if(state->first[dim] == n_samples) {
        std::cout << "Cannot take Gibbs step in empty dimension." << std::endl;
        std::cout << "Exiting lazy_gibbs()" << std::endl;
        return;
    }

    // Copy current state into _dim_ws
    _dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_dim_ws)
    );

    _lazy_gibbs_inner(_dim_ws, dim);

    std::cout << "Exiting lazy_gibbs()" << std::endl;
}


void bridgesamp::BridgingSampler::lazy_gibbs_choose_dim() {
    std::cout << "Entering lazy_gibbs_choose_dim()" << std::endl;

    if(state_rank == n_dim) {
        std::cout << "Cannot take a Gibbs step at the top level of the hierarchy."
                  << std::endl;
        std::cout << "Exiting lazy_gibbs_choose_dim()" << std::endl;
        return;
    }

    // Choose which dimension to step in
    std::uniform_int_distribution<uint16_t> d(0, n_dim-state_rank-1);
    uint16_t i_pick = d(r);
    
    std::cout << "Chose non-empty dimension #" << i_pick << std::endl;

    int32_t i_nonempty = -1;
    uint16_t dim = 0;
    for(auto it=state->first.begin(); i_nonempty != i_pick; ++it, dim++) {
        if(*it != n_samples) {
            i_nonempty++;
        }
    }

    std::cout << " -> dim = " << dim << std::endl;

    lazy_gibbs(dim);

    std::cout << "Exiting lazy_gibbs_choose_dim()" << std::endl;
}


bridgesamp::NodeMap::const_iterator bridgesamp::BridgingSampler::cbegin() const {
    return node.cbegin();
}


bridgesamp::NodeMap::const_iterator bridgesamp::BridgingSampler::cend() const {
    return node.cend();
}


uint16_t bridgesamp::BridgingSampler::get_n_empty(
        const std::vector<uint16_t>& samp)
{
    uint16_t n_empty = 0;
    for(auto s : samp) {
        if(s == n_dim) {
            n_empty++;
        }
    }
    return n_empty;
}


double bridgesamp::BridgingSampler::get_lnp0_of_rank(uint16_t rank) {
    return logp0 + rank * log_n_samples;
}


bridgesamp::NodeMap::iterator bridgesamp::BridgingSampler::get_node(
        const std::vector<uint16_t>& samp)
{
    std::cout << "Entering get_node()" << std::endl;

    // Find node
    auto it = node.find(samp);

    // Get rank of node
    uint16_t n_empty = get_n_empty(samp);
    std::cout << "# empty: " << n_empty << std::endl;

    // Create node if it doesn't exist
    if(it == node.end()) {
        std::cout << "Creating node";
        for(auto s : samp) {
            std::cout << " " << s;
        }
        std::cout << std::endl;

        // Calculate probability of new node
        double logp;
        if(n_empty == 0) {
            // Evaluate rank-0 nodes exactly
            logp = eval_node(samp);
            std::cout << "Evaluated log(p) exactly: " << logp << std::endl;
            
            // Insert the node, and get iterator to it
            it = node.insert(
                std::make_pair(samp, bridgesamp::Node{std::exp(logp), logp})
            ).first;

            // Propagate change in log(p) upward
            percolate_up(it);
        } else {
            // For higher-rank nodes, assume constant probability for
            // all rank-0 child nodes. Probability = const * (# children),
            // where # of children = (# of samples)^rank.
            //logp = logp0 + n_empty * log_n_samples;
            logp = get_lnp0_of_rank(n_empty);
            std::cout << "Set log(p) to default: " << logp << std::endl;
            
            // Insert the node, and get iterator to it
            it = node.insert(
                std::make_pair(samp, bridgesamp::Node{std::exp(logp), logp})
            ).first;
        }
    }

    std::cout << "Exiting get_node()" << std::endl;

    return it;
}


double bridgesamp::add_logs(double loga, double logb) {
    // Numerically stable way to add <a> and <b>, where their logarithms
    // are taken as input, and log(a+b) is given as output.
    if(loga > logb) {
        return loga + std::log(1. + std::exp(logb-loga));
    } else {
        return logb + std::log(1. + std::exp(loga-logb));
    }
}


double bridgesamp::subtract_logs(double loga, double logb) {
    // Numerically stable way to subtract <b> from <a>, where their logarithms
    // are taken as input, and log(a-b) is given as output. If b>a, then
    // returns -infinity.
    if(loga > logb) {
        return loga + std::log(1. - std::exp(logb-loga));
    } else {
        return -std::numeric_limits<double>::infinity();
    }
}


void bridgesamp::BridgingSampler::transition_backward() {
    std::cout << "Entering transition_backward()" << std::endl;
    
    std::cout << "Starting state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;

    if(state_rank == n_samples) {
        std::cout << "Already at highest level in hierarchy." << std::endl
                  << "Exiting transition_backward()" << std::endl;
        return;
    }
    
    // Find non-empty dimensions
    get_nonempty_dims(state, _dim_ws);
    uint16_t n_nonempty = _dim_ws.size();
    std::cout << n_nonempty << " non-empty dimensions" << std::endl;

    // Choose a dimension to blank
    std::uniform_int_distribution<uint16_t> d(0, n_nonempty-1);
    uint16_t idx = _dim_ws[d(r)];
    
    std::cout << "Blanking dimension " << idx << std::endl;

    // Copy current state into _dim_ws
    _dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_dim_ws)
    );
    
    // Blank the chosen dimension and transition
    _dim_ws[idx] = n_samples;
    state = get_node(_dim_ws);
    state_rank++;
    
    std::cout << "New state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    
    std::cout << "Exiting transition_backward()" << std::endl;
}


void bridgesamp::BridgingSampler::transition_forward() {
    std::cout << "Entering transition_forward()" << std::endl;
    
    std::cout << "Starting state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;

    if(state_rank == 0) {
        std::cout << "Already at lowest level in hierarchy." << std::endl
                  << "Exiting transition_forward()" << std::endl;
        return;
    }
    
    // Find empty dimensions
    get_empty_dims(state, _dim_ws);
    uint16_t n_empty = _dim_ws.size();
    std::cout << n_empty << " empty dimensions" << std::endl;

    // Choose a dimension to fill
    std::uniform_int_distribution<uint16_t> d(0, n_empty-1);
    uint16_t idx = _dim_ws[d(r)];
    
    std::cout << "Filling dimension " << idx << std::endl;

    // Copy current state into _dim_ws
    _dim_ws.clear();
    std::copy(
        state->first.begin(),
        state->first.end(),
        std::back_inserter(_dim_ws)
    );
    
    // Fill the chosen dimension with zero
    _dim_ws[idx] = 0;
    state_rank--;

    // Take a Gibbs step
    _lazy_gibbs_inner(_dim_ws, idx);
    
    std::cout << "New state:";
    for(auto s : state->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;
    
    std::cout << "Exiting transition_forward()" << std::endl;
}


void bridgesamp::BridgingSampler::percolate_up(const NodeMap::iterator& n) {
    std::cout << "Entering percolate_up()" << std::endl;
    
    std::cout << "Starting from state:";
    for(auto s : n->first) {
        std::cout << " " << s;
    }
    std::cout << std::endl;

    // Find non-empty dimensions
    get_nonempty_dims(n, _dim_ws);
    std::cout << _dim_ws.size() << " non-empty dimensions" << std::endl;

    // Determine change in probability to propagate
    double dlogp; // log(|dp|)
    bool positive; // True if change is positive
    if(n->second.logp >= logp0) {
        dlogp = subtract_logs(n->second.logp, logp0);
        positive = true;
    } else {
        dlogp = subtract_logs(logp0, n->second.logp);
        positive = false;
    }

    std::cout << "dlogp = " << dlogp << " (";
    if(positive) {
        std::cout << "+";
    } else {
        std::cout << "-";
    }
    std::cout << ")" << std::endl;

    // Loop through higher ranks in hierarchy
    std::vector<unsigned int> idx;
    std::vector<uint16_t> node_key;
    idx.reserve(n_dim);
    node_key.reserve(n_dim);

    for(int rank=1; rank<=_dim_ws.size(); rank++) {
        // Blank out every combination of <rank> entries, obtaining
        // keys to all parent nodes
        CombinationGenerator cgen(_dim_ws.size(), rank);
        while(cgen.next(idx)) {
            std::cout << "idx =";
            for(auto i : idx) {
                std::cout << " " << i;
            }
            std::cout << std::endl;

            // Copy original key into node_key
            node_key.clear();
            std::copy(
                n->first.begin(),
                n->first.end(),
                std::back_inserter(node_key)
            );
            // Blank selected dimensions
            for(auto i : idx) {
                node_key[i] = n_dim;
            }
            std::cout << "node_key =";
            for(auto i : node_key) {
                std::cout << " " << i;
            }
            std::cout << std::endl;

            auto it = get_node(node_key); // Iterator to parent node

            // Update probability of parent node
            if(positive) {
                it->second.logp = add_logs(it->second.logp, dlogp);
            } else {
                it->second.logp = subtract_logs(it->second.logp, dlogp);
            }
        }
    }
    
    std::cout << "Exiting percolate_up()" << std::endl;
}


double bridgesamp::BridgingSampler::n_children(uint16_t order) {
    return pow((double)(n_samples+1), (double)order);
}


void bridgesamp::BridgingSampler::get_nonempty_dims(
        const NodeMap::iterator& n,
        std::vector<uint16_t>& dims_out)
{
    dims_out.clear();
    dims_out.reserve(n_dim);

    for(int i=0; i<n_dim; i++) {
        if(n->first[i] != n_dim) {
            dims_out.push_back(i);
        }
    }
}


void bridgesamp::BridgingSampler::get_empty_dims(
        const NodeMap::iterator& n,
        std::vector<uint16_t>& dims_out)
{
    dims_out.clear();
    dims_out.reserve(n_dim);

    for(int i=0; i<n_dim; i++) {
        if(n->first[i] == n_dim) {
            dims_out.push_back(i);
        }
    }
}



/*
 *  Combination generator - cycles through combinations (n choose r)
 */

bridgesamp::CombinationGenerator::CombinationGenerator(
        unsigned int n, unsigned int r)
    : r(r), n(n), mask(n, false), finished(false)
{
    assert(n >= r);
    std::fill(mask.end()-r, mask.end(), true);
}


void bridgesamp::CombinationGenerator::reset() {
    std::fill(mask.begin(), mask.end()-r-1, false);
    std::fill(mask.end()-r, mask.end(), true);
    finished = false;
}


bool bridgesamp::CombinationGenerator::next(std::vector<unsigned int>& out_idx) {
    if(finished) { return false; }
    
    out_idx.resize(r);
    //assert(out_idx.size() >= r);

    unsigned int k = 0;
    for(unsigned int i=0; i<n; i++) {
        if(mask[i]) {
            out_idx[k] = i;
            k++;
        }
    }

    finished = !std::next_permutation(mask.begin(), mask.end());

    return true;
}

