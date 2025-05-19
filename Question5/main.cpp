#include <iostream>
#include <vector>
#include <string>
#include <memory>  // For smart pointers
#include <cmath>   // For fabs()
#include <iomanip> // For output formatting
#include <stdexcept> // For exceptions

/**
 * @class Node
 * @brief Represents a node in a phylogenetic tree with age and branching structure
 * 
 * Each node stores:
 * - Taxonomic name (empty for internal nodes)
 * - Absolute age (time before present)
 * - Child nodes with their branch lengths
 */
class Node {
public:
    std::string name;  // Node name (empty for internal nodes)
    double age;       // Absolute age of the node
    std::vector<std::pair<double, std::shared_ptr<Node>>> children; // Child nodes with branch lengths

    /**
     * @brief Constructor initializing node name and age
     * @param name_ The name of the node (empty for internal nodes)
     * @param age_ The absolute age of the node (time before present)
     */
    Node(const std::string& name_, double age_) : name(name_), age(age_) {}

    /**
     * @brief Adds a child node with automatically calculated branch length
     * @param child Shared pointer to the child node
     * @throws std::invalid_argument if branch length would be non-positive
     */
    void add_child(std::shared_ptr<Node> child) {
        double branch_length = age - child->age;
        if (branch_length <= 0) {
            throw std::invalid_argument(
                "Branch length must be positive (parent age must be > child age)");
        }
        children.push_back({branch_length, child});
    }
};

/**
 * @brief Helper function for consistent numeric output formatting
 * @param age The age value to print
 * 
 * Prints ages with 3 decimal places, but removes trailing .000 for whole numbers
 */
void print_age(double age) {
    std::cout << std::fixed << std::setprecision(3);
    if (age == std::floor(age)) {
        std::cout << static_cast<int>(age);
    } else {
        std::cout << age;
    }
}

/**
 * @brief Performs post-order traversal of the tree
 * @param node Current node being processed
 * 
 * Prints node information in post-order (children before parent)
 * Format: Node (age=X), name=Y -> #children = Z
 */
void post_order(const std::shared_ptr<Node>& node) {
    // First process all children recursively
    for (const auto& [branch_length, child] : node->children) {
        post_order(child);
    }
    
    // Then print current node information
    std::cout << "Node (age=";
    print_age(node->age);
    std::cout << ")";
    
    if (!node->name.empty()) {
        std::cout << ", name=" << node->name;
    }
    
    std::cout << " -> #children = " << node->children.size() << std::endl;
}

/**
 * @brief Recursive helper function for tree validation
 * @param node Current node being validated
 * @param parent Parent node (nullptr for root)
 * @return true if subtree is valid, false otherwise
 * 
 * Performs the following checks:
 * 1. Node age is non-negative
 * 2. Tip nodes have age = 0
 * 3. Parent age > child age
 * 4. Branch lengths are positive
 * 5. Stored branch length matches age difference
 */
bool verify_helper(const std::shared_ptr<Node>& node, const Node* parent = nullptr) {
    // Check 1: Node age must be non-negative
    if (node->age < 0) {
        std::cerr << "ERROR: Negative age detected at node " << node->name 
                  << " (age = " << node->age << ")\n";
        return false;
    }

    // Check 2: Tip nodes must have age = 0
    if (node->children.empty() && node->age != 0.0) {
        std::cerr << "ERROR: Tip node " << node->name 
                  << " must have age = 0 (found " << node->age << ")\n";
        return false;
    }

    // Check 3: Parent must be older than child (except root)
    if (parent != nullptr && node->age >= parent->age) {
        std::cerr << "ERROR: Invalid age relationship - child " << node->name
                  << " (age=" << node->age << ") >= parent (age=" 
                  << parent->age << ")\n";
        return false;
    }

    // Validate all child nodes
    for (const auto& [branch_length, child] : node->children) {
        // Print branch information for debugging
        std::cout << "Parent age: ";
        print_age(node->age);
        std::cout << " -> Child age: ";
        print_age(child->age);
        std::cout << " | Branch length: ";
        print_age(branch_length);
        std::cout << std::endl;

        // Check 4: Branch length must be positive
        if (branch_length <= 0.0) {
            std::cerr << "ERROR: Non-positive branch length ("
                      << branch_length << ") between "
                      << node->name << " and " << child->name << "\n";
            return false;
        }

        // Check 5: Stored branch length must match age difference
        double expected_length = node->age - child->age;
        if (std::fabs(branch_length - expected_length) > 1e-9) {
            std::cerr << "ERROR: Branch length mismatch. Expected "
                      << expected_length << " but found " << branch_length
                      << " between " << node->name << " and " << child->name << "\n";
            return false;
        }

        // Recursively validate child subtree
        if (!verify_helper(child, node.get())) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Public interface for tree validation
 * @param root Root node of the tree to validate
 * @return true if tree is valid, false otherwise
 * 
 * Initiates validation and prints final result message
 */
bool verify_tree(const std::shared_ptr<Node>& root) {
    std::cout << "\nVerifying tree structure:\n";
    bool valid = verify_helper(root);
    
    if (valid) {
        std::cout << "\nTree verification PASSED: structurally and numerically valid.\n";
    } else {
        std::cout << "\nTree verification FAILED.\n";
    }
    
    return valid;
}

/**
 * @brief Builds the example tree from the problem statement
 * @return Shared pointer to the root node
 * 
 * Constructs the following tree structure:
 *        root(0.2)
 *        /      \
 *     n2(0.1)   t5(0)
 *    /   |   \
 * n1(0.05) t3(0) t4(0)
 * /   \
 * t1(0) t2(0)
 */
std::shared_ptr<Node> build_example_tree() {
    // Create tip nodes (age = 0)
    auto t1 = std::make_shared<Node>("t1", 0.0);
    auto t2 = std::make_shared<Node>("t2", 0.0);
    auto t3 = std::make_shared<Node>("t3", 0.0);
    auto t4 = std::make_shared<Node>("t4", 0.0);
    auto t5 = std::make_shared<Node>("t5", 0.0);

    // Build tree structure with automatic branch length calculation
    auto node_0_05 = std::make_shared<Node>("", 0.05);
    node_0_05->add_child(t1);  // branch length = 0.05 - 0.0 = 0.05
    node_0_05->add_child(t2);  // branch length = 0.05 - 0.0 = 0.05

    auto node_0_10 = std::make_shared<Node>("", 0.10);
    node_0_10->add_child(node_0_05); // branch length = 0.10 - 0.05 = 0.05
    node_0_10->add_child(t3);        // branch length = 0.10 - 0.0 = 0.10
    node_0_10->add_child(t4);        // branch length = 0.10 - 0.0 = 0.10

    auto root = std::make_shared<Node>("", 0.20);
    root->add_child(node_0_10); // branch length = 0.20 - 0.10 = 0.10
    root->add_child(t5);        // branch length = 0.20 - 0.0 = 0.20

    return root;
}

/**
 * @brief Main function demonstrating tree construction and validation
 */
int main() {
    try {
        // Build the example tree
        auto root = build_example_tree();

        // Print post-order traversal
        std::cout << "Post-order traversal (age and child count):\n";
        post_order(root);

        // Validate the tree structure
        verify_tree(root);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}