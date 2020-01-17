package kungfu

import (
	kb "github.com/lsds/KungFu/srcs/go/kungfubase"
	"github.com/lsds/KungFu/srcs/go/plan"
)

type partitionStrategy func(plan.PeerList) []strategy

var partitionStrategies = map[kb.Strategy]partitionStrategy{
	kb.Star:                    createStarStrategies,
	kb.Clique:                  createCliqueStrategies,
	kb.Ring:                    createRingStrategies,
	kb.Tree:                    createTreeStrategies,
	kb.BinaryTree:              createBinaryTreeStrategies,
	kb.BinaryTreeStar:          createBinaryTreeStarStrategies,
	kb.BinaryTreePrimaryBackup: CreatePrimaryBackupStrategies,
	kb.PrimaryBackupTesting:    CreatePrimaryBackupStrategiesTesting,
}

func simpleSingleGraphStrategy(bcastGraph *plan.Graph) []strategy {
	r := plan.GenDefaultReduceGraph(bcastGraph)
	b := bcastGraph
	// r.Debug()
	// b.Debug()

	return []strategy{
		{
			reduceGraph: r,
			bcastGraph:  b,
		},
	}
}

func createStarStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenStarBcastGraph(len(peers), defaultRoot)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createTreeStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenTree(peers)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createBinaryTreeStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenBinaryTree(0, len(peers))
	return simpleSingleGraphStrategy(bcastGraph)
}

func CreatePrimaryBackupStrategies(peers plan.PeerList) []strategy {
	numPrimaries := len(peers) / 2
	numBackups := len(peers) / 2
	bcastGraph := plan.GenBinaryTreePrimaryBackup(numPrimaries, numBackups)
	return simpleSingleGraphStrategy((bcastGraph))
}

func CreatePrimaryBackupStrategiesTesting(peers plan.PeerList) []strategy {
	// method for inserting custom strategies for testing purposes.
	return createStarPrimaryBackupStrategies(peers)
}

func createBinaryTreeStarStrategies(peers plan.PeerList) []strategy {
	bcastGraph := plan.GenBinaryTreeStar(peers)
	return simpleSingleGraphStrategy(bcastGraph)
}

func createCliqueStrategies(peers plan.PeerList) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		bcastGraph := plan.GenStarBcastGraph(k, r)
		reduceGraph := plan.GenDefaultReduceGraph(bcastGraph)
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func createRingStrategies(peers plan.PeerList) []strategy {
	k := len(peers)
	var ss []strategy
	for r := 0; r < k; r++ {
		reduceGraph, bcastGraph := plan.GenCircularGraphPair(k, r)
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func createStarPrimaryBackupStrategies(peers plan.PeerList) []strategy {
	k := len(peers)
	reduceGraph, bcastGraph := plan.GenStarPrimaryBackupGraphPair(0, k-1, 1)
	var ss []strategy
	ss = append(ss, strategy{
		reduceGraph: reduceGraph,
		bcastGraph:  bcastGraph,
	})
	// reduceGraph.Debug()
	// bcastGraph.Debug()
	return ss
}

func autoSelect(peers plan.PeerList) kb.Strategy {
	m := make(map[uint32]int)
	for _, p := range peers {
		m[p.IPv4]++
	}
	if len(m) == 1 {
		return kb.Star
	}
	return kb.BinaryTreeStar
}
