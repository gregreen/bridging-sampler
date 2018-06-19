
#include <iostream>
#include "bridging_sampler.h"

void print_state(const bridgesamp::BridgingSampler& sampler) {
    std::cout << "* ";

    for(auto s : sampler.get_state()) {
        if(s == sampler.get_n_samples()) {
            std::cout << "- ";
        } else {
            std::cout << s << " ";
        }
    }

    std::cout << "(" << sampler.get_state_rank() << ")"
              << std::endl << std::endl;

    for(auto it=sampler.cbegin(); it!=sampler.cend(); ++it) {
        for(auto s : it->first) {
            if(s == sampler.get_n_samples()) {
                std::cout << "- ";
            } else {
                std::cout << s << " ";
            }
        }
        std::cout << ": " << std::exp(it->second.logp) << std::endl;
    }
}

void binary_labeling_toy() {

    auto f_prob = [](const std::vector<uint16_t>& s) -> double {
        if(s[0] == s[1]) {
            return -100.;  // Zero probability for (0,0) and (1,1)
        } else {
            return 0.;     // Probability=1 for (1,0) and (0,1)
        }
    };

    // Set up the sampler, providing a function that calculates
    // log(probability) for each state
    bridgesamp::BridgingSampler sampler(
        2, 2, // # of dimensions, # of samples per dimension
        f_prob
    );
    
    sampler.randomize_state();

    // Burn-in
    for(int n=0; n<1000; n++) {
        sampler.step();
    }

    // Store the number of visits to each state
    std::unordered_map<std::vector<uint16_t>, uint32_t, bridgesamp::VectorHasher> n_visits;

    // Sample
    for(int n=0; n<1000; n++) {
        sampler.step();
        n_visits[sampler.get_state()]++;
    }
    
    // Print out the number of visits to each state
    std::cout << "# of visits:"
              << std::endl;
    
    for(auto& v : n_visits) {
        for(auto s : v.first) {
            if(s == sampler.get_n_samples()) {
                std::cout << "- ";
            } else {
                std::cout << s << " ";
            }
        }
        std::cout << ": " << v.second << std::endl;
    }
}


void toy() {
    double penalty = 5.;
    auto f_prob = [penalty](const std::vector<uint16_t>& s) -> double {
        double logp = 0.;
        for(int j=0; j<s.size()-1; j++) {
            for(int k=j+1; k<s.size(); k++) {
                if(s[j] != s[k]) {
                    logp -= penalty;
                }
            }
        }
        return logp;
    };

    auto f_cond = [](uint16_t dim, const std::vector<uint16_t>& s, std::vector<double>& logp) -> void {
        std::fill(logp.begin(), logp.end(), -100.);

        // Bail if not all samples (other than in the requested dimension) are the same
        if(s.size() > 2) {
            uint16_t i = 0;
            for(auto x : s) {
                if(i++ == dim) { continue; }
                if(x != s[0]) {
                    return;
                }
            }
        }
        
        logp[s[dim]] = 0.;
    };

    bridgesamp::BridgingSampler sampler(
        3, 25, f_prob
    );

    sampler.set_logp0(-0.5 * sampler.get_n_dim() * penalty * 0.25);
    
    sampler.randomize_state();

    std::cout << "Initial sampler state:" << std::endl;
    print_state(sampler);
    std::cout << std::endl;
    //chain.push_back(sampler.get_state());

    for(int n=0; n<1000000; n++) {
        sampler.step();
        //std::cout << std::endl;
        //print_state(sampler);
        //std::cout << std::endl;
        //chain.push_back(sampler.get_state());
    }
    
    //std::vector<std::vector<uint16_t>> chain;
    std::unordered_map<std::vector<uint16_t>, uint32_t, bridgesamp::VectorHasher> n_visits;

    for(int n=0; n<1000000; n++) {
        sampler.step();
        //std::cout << std::endl;
        //print_state(sampler);
        //std::cout << std::endl;
        //chain.push_back(sampler.get_state());
        n_visits[sampler.get_state()]++;
    }

    //std::cout << std::endl
    //          << "chain:"
    //          << std::endl;
    //for(auto& c : chain) {
    //    for(auto s : c) {
    //        std::cout << s << " ";
    //    }
    //    std::cout << std::endl;
    //}
    
    //std::cout << "Final sampler state:" << std::endl;
    //print_state(sampler);
    //std::cout << std::endl;
    
    //std::cout << std::endl
    //          << "# of visits:"
    //          << std::endl;
    //for(auto& v : n_visits) {
    //    for(auto s : v.first) {
    //        if(s == sampler.get_n_samples()) {
    //            std::cout << "- ";
    //        } else {
    //            std::cout << s << " ";
    //        }
    //    }
    //    std::cout << ": " << v.second << std::endl;
    //}

    std::cout << std::endl
              << "Target space:" << std::endl;
    
    auto is_target = [&sampler](const std::vector<uint16_t>& s) -> bool {
        for(auto x : s) {
            if(x == sampler.get_n_samples()) {
                return false;
            }
        }
        return true;
    };

    //for(auto& v : n_visits) {
    //    if(is_target(v.first)) {
    //        for(auto s : v.first) {
    //            std::cout << s << " ";
    //        }
    //        std::cout << ": " << v.second << std::endl;
    //    }
    //}

    // Sort elements by # of visits
    typedef std::pair<std::vector<uint16_t>, uint32_t> node_t;
    //typedef std::function<bool(const node_t&, const node_t&)> node_comparator_t;
    std::vector<node_t> n_visits_sorted(n_visits.begin(), n_visits.end());
    std::sort(
        n_visits_sorted.begin(),
        n_visits_sorted.end(),
        [](const node_t& n1, const node_t& n2) -> bool {
            return n1.second > n2.second;
        }
    );

    uint32_t n_visits_tot = std::accumulate(
        n_visits_sorted.begin(),
        n_visits_sorted.end(),
        0.,
        [is_target](uint32_t v, const node_t& n) {
            if(is_target(n.first)) {
                return v + n.second;
            } else {
                return v;
            }
        }
    );

    uint32_t n_cumulative = 0;

    for(auto& v : n_visits_sorted) {
        if(is_target(v.first)) {
            for(auto s : v.first) {
                std::cout << s << " ";
            }
            std::cout << ": " << v.second << std::endl;

            n_cumulative += v.second;
            if((n_cumulative > 0.99 * n_visits_tot) ||
               (v.second < 0.01 * n_visits_sorted.at(0).second))
            {
               break;
            }
        }
    }

    std::cout << "All others: "
              << n_visits_tot - n_cumulative
              << std::endl;

    std::cout << std::endl
              << 100. * sampler.fill_factor()
              << "% of nodes explored."
              << std::endl;

}


int main() {
    toy();
    //binary_labeling_toy();

    return 0;
}
