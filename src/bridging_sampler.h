
#ifndef __BRIDGING_SAMPLER_H__
#define __BRIDGING_SAMPLER_H__

#include <cstdint>
#include <vector>
#include <map>
#include <random>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>

#include <iostream>


namespace bridgesamp {

struct Node {
    double p, logp;
};


typedef std::map<std::vector<uint16_t>, Node> NodeMap;


class BridgingSampler {
private:
    uint16_t n_dim;     // # of integers to define a state
    uint16_t n_samples; // # of samples per dimension
    
    NodeMap node; // All visited nodes
    NodeMap::iterator state; // Points to current node
    uint16_t state_rank; // Rank of current state
    
    // Function that returns log(p) of node
    std::function<double(const std::vector<uint16_t>&)> eval_node;

    double p0, logp0; // Assumed value of p (or log(p)) for unexplored nodes
    double log_n_samples;

    // Sampling parameters
    std::vector<double> b_prob; // Probability of backward transition at each rank
    std::vector<double> f_prob; // Probability of forward transition at each rank

    // Random numbers
    std::mt19937 r;
    std::uniform_int_distribution<uint16_t> r_dim;    // Draw a random dimension
    std::uniform_int_distribution<uint16_t> r_samp;   // Draw a random sample
    std::uniform_real_distribution<double> r_uniform; // Draw from U(0,1)

    // Workspace
    std::vector<uint16_t> _samp_ws; // Workspace with capacity equal to n_samples
    std::vector<uint16_t> _dim_ws;  // Workspace with capacity equal to n_dim
    
    std::vector<double> _samp_ws_dbl; // Workspace with capacity equal to n_samples

    // Navigation
    //std::map<std::vector<uint16_t>, Node>::iterator up(
    //        std::map<std::vector<uint16_t>, Node>::iterator s,
    //        int
    
    // Returns an iterator to the node specified by the given sample numbers
    NodeMap::iterator get_node(const std::vector<uint16_t>& samp);

    // # of children of node of given order
    double n_children(uint16_t order);
    
    // # of dimensions that are empty in key
    uint16_t get_n_empty(const std::vector<uint16_t>& samp);

    // Sampling routines
    void percolate_up(const NodeMap::iterator& n);
    
    void _lazy_gibbs_inner(std::vector<uint16_t>& starting_state,
                           uint16_t dim);

    // Get dimensions which are empty
    void get_empty_dims(const NodeMap::iterator& n,
                        std::vector<uint16_t>& dims_out);
    
    // Get dimensions which are set (non-empty)
    void get_nonempty_dims(const NodeMap::iterator& n,
                           std::vector<uint16_t>& dims_out);

    double get_lnp0_of_rank(uint16_t rank);

public:
    BridgingSampler(uint16_t n_dim,
                    uint16_t n_samples,
                    std::function<double(const std::vector<uint16_t>&)> logp_node);

    void step(); // Choose and execute a step type

    void randomize_state(); // Jump to randomly chosen base state
    
    // Take a Gibbs step in the specified dimension, without evaluating
    // unexplored states.
    void lazy_gibbs(uint16_t dim);

    // Take a Gibbs step, choosing the dimension randomly
    void lazy_gibbs_choose_dim();
    
    // Transition to a randomly selected state one level up in the hierarchy
    void transition_backward();
    
    // Transition to a randomly selected state one level down in the hierarchy
    void transition_forward();

    NodeMap::const_iterator cbegin() const;
    NodeMap::const_iterator cend() const;

    uint16_t get_n_dim() const;
    uint16_t get_n_samples() const;

    const std::vector<uint16_t>& get_state() const;
};


class CombinationGenerator {
    // Cycles through combinations of elements. Generates
    // n choose r combinations.
public:
    // n choose r
    CombinationGenerator(unsigned int n, unsigned int r);

    // Set <out_idx> to the next combination. Returns false
    // if the final combination has already been reached.
    bool next(std::vector<unsigned int>& out_idx);

    // Reset the combination generator to the beginning
    void reset();

private:
    std::vector<bool> mask; // True for indices that will be selected
    unsigned int n, r;
    bool finished; // True if last combination has been reached
};


// Misc functions
double add_logs(double loga, double logb);
double subtract_logs(double loga, double logb);


}

#endif // __BRIDGING_SAMPLER_H__
