package main

import "github.com/lsds/KungFu/srcs/go/utils"

import "C"

//export GoKungfuResizeCluster
func GoKungfuResizeCluster(pCkpt *C.char, size int, pChanged, pKeep *C.char) int {
	ckpt := C.GoString(pCkpt)
	changed, keep, err := kungfu.ResizeCluster(ckpt, size)
	if err != nil {
		utils.ExitErr(err)
	}
	*pChanged = boolToChar(changed)
	*pKeep = boolToChar(keep)
	return 0
}

//export GoKungfuReshapeStrategy
func GoKungfuReshapeStrategy(step int, pStrategyChanged *C.char) int {
	StrategyChanged, err := kungfu.ReshapeStrategy(step)
	if err != nil {
		utils.ExitErr(err)
	}
	*pStrategyChanged = boolToChar(StrategyChanged)
	return 0
}
