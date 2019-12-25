package kungfubase

import "errors"

// #include "kungfu/strategy.h"
import "C"

type Strategy C.KungFu_Strategy

const (
	Star                    Strategy = C.KungFu_Star
	Ring                    Strategy = C.KungFu_Ring
	Clique                  Strategy = C.KungFu_Clique
	Tree                    Strategy = C.KungFu_Tree
	BinaryTree              Strategy = C.KungFu_BinaryTree
	BinaryTreeStar          Strategy = C.KungFu_BinaryTreeStar
	Auto                    Strategy = C.KungFu_AUTO
	BinaryTreePrimaryBackup Strategy = C.KungFu_BinaryTreePrimaryBackup
	PrimaryBackupTesting  	Strategy = C.KungFu_PrimaryBackupTesting
)

const DefaultStrategy = BinaryTreeStar

var (
	StrategyNamesMap = map[Strategy]string{
		Star:                    `STAR`,
		Ring:                    `RING`,
		Clique:                  `CLIQUE`,
		Tree:                    `TREE`,
		BinaryTree:              `BINARY_TREE`,
		BinaryTreeStar:          `BINARY_TREE_STAR`,
		Auto:                    `AUTO`,
		BinaryTreePrimaryBackup: `BINARY_TREE_PRIMARY_BACKUP`,
		PrimaryBackupTesting: `PRIMARY_BACKUP_TESTING`,
	}
)

var StrategyNamesArray = []Strategy{Star, Ring, Clique, Tree, BinaryTree, BinaryTreeStar}

func StrategyNames() []string {
	var names []string
	for _, name := range StrategyNamesMap {
		names = append(names, name)
	}
	return names
}

func (s Strategy) String() string {
	return StrategyNamesMap[s]
}

// Set implements flags.Value::Set
func (s *Strategy) Set(val string) error {
	value, err := ParseStrategy(val)
	if err != nil {
		return err
	}
	*s = *value
	return nil
}

var errInvalidStrategy = errors.New("invalid strategy")

func ParseStrategy(s string) (*Strategy, error) {
	for k, v := range StrategyNamesMap {
		if s == v {
			return &k, nil
		}
	}
	return nil, errInvalidStrategy
}
