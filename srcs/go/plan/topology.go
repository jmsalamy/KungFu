package plan

func getLocalMasters(peers PeerList) ([]int, map[uint32]int) {
	var masters []int
	hostMaster := make(map[uint32]int)
	for rank, p := range peers {
		if _, ok := hostMaster[p.IPv4]; !ok {
			hostMaster[p.IPv4] = rank
			masters = append(masters, rank)
		}
	}
	return masters, hostMaster
}

func GenTree(peers PeerList) *Graph {
	g := NewGraph(len(peers))
	masters, hostMaster := getLocalMasters(peers)
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank {
			g.AddEdge(master, rank)
		}
	}
	if len(masters) > 1 {
		for _, rank := range masters[1:] {
			g.AddEdge(masters[0], rank)
		}
	}
	return g
}

func GenDefaultReduceGraph(g *Graph) *Graph {
	g0 := g.Reverse()
	k := len(g.Nodes)
	for i := 0; i < k; i++ {
		g0.AddEdge(i, i)
	}
	return g0
}

func GenBinaryTree(r, k int) *Graph {
	g := NewGraph(k)
	for i := r; i < k; i++ {
		if j := i*2 + 1; j < k {
			g.AddEdge(i, j)
		}
		if j := i*2 + 2; j < k {
			g.AddEdge(i, j)
		}
	}
	return g
}

// GenBinaryTreePrimaryBackup create a simple primary backup strategy
func GenBinaryTreePrimaryBackup(numPrimaries, numBackups int) *Graph {
	// TODO: refactor in terms of GenBinaryTree method
	g := NewGraph(numPrimaries + numBackups)
	for i := 0; i < numPrimaries; i++ {
		if j := i*2 + 1; j < numPrimaries {
			g.AddEdge(i, j)
		}
		if j := i*2 + 2; j < numPrimaries {
			g.AddEdge(i, j)
		}
	}

	for i := numPrimaries; i < numBackups+numPrimaries; i++ {
		if j := i*2 + 1 - numPrimaries; j < numBackups+numPrimaries {
			g.AddEdge(i, j)
		}
		if j := i*2 + 2 - numPrimaries; j < numBackups+numPrimaries {
			g.AddEdge(i, j)

		}
	}
	return g
}

// func GenCustomTreePrimaryBackup()

func GenBinaryTreeStar(peers PeerList) *Graph {
	g := NewGraph(len(peers))
	masters, hostMaster := getLocalMasters(peers)
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank {
			g.AddEdge(master, rank)
		}
	}
	if k := len(masters); k > 1 {
		for i := 0; i < k; i++ {
			if j := i*2 + 1; j < k {
				g.AddEdge(masters[i], masters[j])
			}
			if j := i*2 + 2; j < k {
				g.AddEdge(masters[i], masters[j])
			}
		}
	}
	return g
}

// GenStarBcastGraph generates a star shape graph with k vertices and centered at vertice r (0 <= r < k)
func GenStarBcastGraph(k, r int) *Graph {
	g := NewGraph(k)
	for i := 0; i < k; i++ {
		if i != r {
			g.AddEdge(r, i)
		}
	}
	return g
}

func GenCircularGraphPair(k, r int) (*Graph, *Graph) {
	g := NewGraph(k)
	for i := 0; i < k; i++ {
		g.AddEdge(i, i)
	}
	b := NewGraph(k)
	for i := 1; i < k; i++ {
		g.AddEdge((r+i)%k, (r+i+1)%k)
		b.AddEdge((r+i-1)%k, (r+i)%k)
	}
	return g, b
}

func GenCircularGraphPairFromConfig(k, reduceEdgeToRemove, bcastEdgeToRemove int, primaries, backups []int) (*Graph, *Graph) {
	r := NewGraph(k)
	b := NewGraph(k)

	for i := 0; i < k; i++ {
		r.AddEdge(i, i)
	}

	var bcastGraphLastNode int
	for i := 0; i < len(primaries); i++ {
		fromNode, toNode := primaries[i], primaries[(i+1)%len(primaries)]
		if i != reduceEdgeToRemove {
			r.AddEdge(fromNode, toNode)

		}
		if i != bcastEdgeToRemove {
			b.AddEdge(fromNode, toNode)
			bcastGraphLastNode = toNode
		}
	}

	// add final bcastGraph edge for pushing values to the disconnected worker
	fromNode := bcastGraphLastNode
	for i := 0; i < len(backups); i++ {
		toNode := backups[i]
		b.AddEdge(fromNode, toNode)
		fromNode = toNode
	}
	// start with a full ring and remove the appropriate 1 edge to create a rooted ring topology for reduce
	// and bcast graph
	return r, b
}

// GenBinaryTreeStarPrimaryBackupGraphPair generates primary-backup strategy based off of BinaryTreeStar Default strategy
func GenBinaryTreeStarPrimaryBackupGraphPair(peers PeerList, numPrimaries, numBackups int) (*Graph, *Graph) {
	// Generate default broadcast graph for BinaryTreeStar
	b := GenBinaryTreeStar(peers)

	// Generate custom reduceGraph to allow for primary-backup strategy
	r := NewGraph(len(peers))
	k := len(r.Nodes)
	numPeers := len(peers) - 1
	// add self loops first

	masters, hostMaster := getLocalMasters(peers)
	for rank, p := range peers {
		if master := hostMaster[p.IPv4]; master != rank && rank != numPeers {
			r.AddEdge(master, rank)
		}
	}
	if k := len(masters); k > 1 {
		for i := 0; i < k; i++ {
			if j := i*2 + 1; j < k && j != numPeers {
				r.AddEdge(masters[i], masters[j])
			}
			if j := i*2 + 2; j < k && j != numPeers {
				r.AddEdge(masters[i], masters[j])
			}
		}
	}
	reduceGraphReversed := r.Reverse()
	for i := 0; i < k; i++ {
		reduceGraphReversed.AddEdge(i, i)
	}
	return reduceGraphReversed, b
}
