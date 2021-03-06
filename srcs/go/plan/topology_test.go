package plan

import "testing"

type edge struct {
	from int
	to   int
}

func isValidGraph(g *Graph) bool {
	k := len(g.Nodes)
	m := make(map[edge]int)
	for i := 0; i < k; i++ {
		if g.Nodes[i].Rank != i {
			return false
		}
		for _, j := range g.Nodes[i].Nexts {
			e := edge{i, j}
			if m[e]++; m[e] > 1 {
				return false
			}
		}
	}
	var n int
	for i := 0; i < k; i++ {
		for _, j := range g.Nodes[i].Prevs {
			n++
			e := edge{j, i}
			if m[e] != 1 {
				return false
			}
		}
	}
	if n != len(m) {
		return false
	}
	return true
}

func isValidTreeWithRoot(g *Graph, root int) bool {
	if !isValidGraph(g) {
		return false
	}
	k := len(g.Nodes)
	p := make(map[int]int)
	for i := 0; i < k; i++ {
		if g.Nodes[i].SelfLoop {
			return false
		}
		for _, j := range g.Nodes[i].Nexts {
			if _, ok := p[j]; ok {
				return false
			}
			p[j] = i
		}
	}
	if len(p) != k-1 {
		return false
	}
	if _, ok := p[root]; ok {
		return false
	}
	return true
}

func Test_trees(t *testing.T) {
	peers := PeerList{
		{3, 9}, // 0
		{3, 8}, // 1
		{2, 7}, // 2 *
		{2, 6}, // 3
		{2, 5}, // 4
		{1, 4}, // 5 *
		{1, 3}, // 6
		{1, 2}, // 7
		{1, 1}, // 8
	}
	if g := GenTree(peers); !isValidTreeWithRoot(g, 0) {
		t.Errorf("tree not generated correctly")
	}
	if g := GenBinaryTree(0, len(peers)); !isValidTreeWithRoot(g, 0) {
		t.Errorf("tree not generated correctly")
	}
	if g := GenBinaryTreeStar(peers); !isValidTreeWithRoot(g, 0) {
		t.Errorf("tree not generated correctly")
	}
}

func Test_PrimaryBackupTree(t *testing.T) {
	if g := GenBinaryTreePrimaryBackup(2, 2); !isValidGraph(g) {
		t.Errorf("primary backup binary tree not generated correctly")
	}
}

func Test_GenCircularGraphPair(t *testing.T) {
	// k := 5
	// // b := GenSimpleRingStrategy(k)
	// fmt.Println("------------------------------")
	// b.Debug()
	t.Errorf("code here")
}
