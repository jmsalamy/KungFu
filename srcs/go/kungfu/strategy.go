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
	config := map[int]bool{
		0: true,
		1: true,
		2: true,
		3: false,
		4: true,
	}
	return createRingStrategiesFromConfig(peers, config)
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

func createRingStrategiesFromConfig(peers plan.PeerList, config map[int]bool) []strategy {
	k := len(peers)
	var ss []strategy

	var primaries []int
	var backups []int

	for i := 0; i < k; i++ {
		if config[i] {
			primaries = append(primaries, i)
		}
		if !config[i] {
			backups = append(backups, i)
		}
	}
	numActive := len(primaries)

	for r := 0; r < k; r++ {
		reduceEdgeToRemove, bcastEdgeToRemove := r%numActive, (r+(numActive-1))%numActive
		reduceGraph, bcastGraph := plan.GenCircularGraphPairFromConfig(k, reduceEdgeToRemove, bcastEdgeToRemove, primaries, backups)
		reduceGraph.Debug()
		bcastGraph.Debug()
		ss = append(ss, strategy{
			reduceGraph: reduceGraph,
			bcastGraph:  bcastGraph,
		})
	}
	return ss
}

func createStarPrimaryBackupStrategies(peers plan.PeerList) []strategy {
	k := len(peers)
	reduceGraph, bcastGraph := plan.GenBinaryTreeStarPrimaryBackupGraphPair(peers, k-1, 1)
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

type Delay struct {
	IterationID int
	NodeID      int
	TimeDelay   int
}

func GenerateConfigFromDelay(k int, delay Delay) map[int]bool {
	config := make(map[int]bool)
	for i := 0; i < k; i++ {
		if delay.NodeID == i {
			config[i] = false
		} else {
			config[i] = true

		}
	}

	return config
}
