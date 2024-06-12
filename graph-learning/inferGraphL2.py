# Copyright (c) 2015, Stanford University. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from snap import *
from cvxpy import *

import math
import multiprocessing
import numpy
from scipy.sparse import lil_matrix
import sys
import time
import builtins
import code
# File format: One edge per line, written as "srcID dstID"
# Commented lines that start with '#' are ignored
# Returns a TGraphVX object with the designated edges and nodes
from snap.snap import TUNGraph


def LoadEdgeList(Filename):
    gvx = TGraphVX()
    nids = set()
    infile = open(Filename, 'r')
    with open(Filename) as infile:
        for line in infile:
            if line.startswith('#'): continue
            [src, dst] = line.split()
            if int(src) not in nids:
                gvx.AddNode(int(src))
                nids.add(int(src))
            if int(dst) not in nids:
                gvx.AddNode(int(dst))
                nids.add(int(dst))
            gvx.AddEdge(int(src), int(dst))
    return gvx


# TGraphVX inherits from the TUNGraph object defined by Snap.py
class TGraphVX(TUNGraph):

    __default_objective = norm(0)
    __default_constraints = []

    # Data Structures
    # ---------------
    # node_objectives  = {int NId : CVXPY Expression}
    # node_constraints = {int NId : [CVXPY Constraint]}
    # edge_objectives  = {(int NId1, int NId2) : CVXPY Expression}
    # edge_constraints = {(int NId1, int NId2) : [CVXPY Constraint]}
    # all_variables = set(CVXPY Variable)
    #
    # ADMM-Specific Structures
    # ------------------------
    # node_variables   = {int NId :
    #       [(CVXPY Variable id, CVXPY Variable name, CVXPY Variable, offset)]}
    # node_values = {int NId : numpy array}
    # node_values points to the numpy array containing the value of the entire
    #     variable space corresponding to then node. Use the offset to get the
    #     value for a specific variable.
    #
    # Constructor
    # If Graph is a Snap.py graph, initializes a SnapVX graph with the same
    # nodes and edges.
    def __init__(self, Graph=None):
        # Initialize data structures
        self.node_objectives = {}
        self.node_variables = {}
        self.node_constraints = {}
        self.edge_objectives = {}
        self.edge_constraints = {}
        self.node_values = {}
        self.all_variables = set()
        self.status = None
        self.value = None

        # Initialize superclass
        nodes = 0
        edges = 0
        if Graph != None:
            nodes = Graph.GetNodes()
            edges = Graph.GetEdges()
        TUNGraph.__init__(self, nodes, edges)

        # Support for constructor with Snap.py graph argument
        if Graph != None:
            for ni in Graph.Nodes():
                self.AddNode(ni.GetId())
            for ei in Graph.Edges():
                self.AddEdge(ei.GetSrcNId(), ei.GetDstNId())

    # Simple iterator to iterator over all nodes in graph. Similar in
    # functionality to Nodes() iterator of PUNGraph in Snap.py.
    def Nodes(self):
        ni = TUNGraph.BegNI(self)
        for i in range(TUNGraph.GetNodes(self)):
            yield ni
            ni.Next()

    # Simple iterator to iterator over all edge in graph. Similar in
    # functionality to Edges() iterator of PUNGraph in Snap.py.
    def Edges(self):
        ei = TUNGraph.BegEI(self)
        for i in range(TUNGraph.GetEdges(self)):
            yield ei
            ei.Next()

    # Adds objectives together to form one collective CVXPY Problem.
    # Option of specifying Maximize() or the default Minimize().
    # Graph status and value properties will also be set.
    # Individual variable values can be retrieved using GetNodeValue().
    # Option to use serial version or distributed ADMM.
    # maxIters optional parameter: Maximum iterations for distributed ADMM.
    def Solve(self, M=Minimize, UseADMM=True, NumProcessors=0, Rho=1.0,
              MaxIters=250, EpsAbs=0.01, EpsRel=0.01, Verbose=False, 
              UseClustering = False, ClusterSize = 1000 ):
        global m_func
        m_func = M

        # Use ADMM if the appropriate parameter is specified and if there
        # are edges in the graph.
        #if builtins.len(SuperNodes) > 0:
        if UseClustering and ClusterSize > 0:
            SuperNodes = self.__ClusterGraph(ClusterSize)
            self.__SolveClusterADMM(M,UseADMM,SuperNodes, NumProcessors, Rho, MaxIters,\
                                     EpsAbs, EpsRel, Verbose)
            return
        if UseADMM and self.GetEdges() != 0:
            self.__SolveADMM(NumProcessors, Rho, MaxIters, EpsAbs, EpsRel,
                             Verbose)
            return
        if Verbose:
            print('Serial ADMM')
        objective = 0
        constraints = []
        # Add all node objectives and constraints
        for ni in self.Nodes():
            nid = ni.GetId()
            objective += self.node_objectives[nid]
            constraints += self.node_constraints[nid]
        # Add all edge objectives and constraints
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            objective += self.edge_objectives[etup]
            constraints += self.edge_constraints[etup]
        # Solve CVXPY Problem
        objective = m_func(objective)
        problem = Problem(objective, constraints)
        try:
            problem.solve()
        except SolverError:
            problem.solve(solver=SCS)
        if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
            problem.solve(solver=SCS)
        # Set TGraphVX status and value to match CVXPY
        self.status = problem.status
        self.value = problem.value
        # Insert into hash to support ADMM structures and GetNodeValue()
        for ni in self.Nodes():
            nid = ni.GetId()
            variables = self.node_variables[nid]
            value = None
            for (varID, varName, var, offset) in variables:
                if var.size[0] == 1:
                    val = numpy.array([var.value])
                else:
                    val = numpy.array(var.value).reshape(-1,)
                if value is None:
                    value = val
                else:
                    value = numpy.concatenate((value, val))
            self.node_values[nid] = value

    """Function to solve cluster wise optimization problem"""
    def __SolveClusterADMM(self,M,UseADMM,superNodes,numProcessors, rho_param, 
                           maxIters, eps_abs, eps_rel,verbose):
        #initialize an empty supergraph
        supergraph = TGraphVX()
        nidToSuperidMap = {}
        edgeToClusterTupMap = {}
        for snid in range(builtins.len(superNodes)):
            for nid in superNodes[snid]:
                nidToSuperidMap[nid] = snid
        """collect the entities for the supergraph. a supernode is a subgraph. a superedge
        is a representation of a graph cut"""
        superEdgeObjectives = {}
        superEdgeConstraints = {}
        superNodeObjectives = {}
        superNodeConstraints = {}
        superNodeVariables = {}
        superNodeValues = {}
        varToSuperVarMap = {}
        """traverse through the list of edges and add each edge's constraint and objective to 
        either the supernode to which it belongs or the superedge which connects the ends 
        of the supernodes to which it belongs"""
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            supersrcnid,superdstnid = nidToSuperidMap[etup[0]],nidToSuperidMap[etup[1]]
            if supersrcnid != superdstnid:    #the edge is a part of the cut
                if supersrcnid > superdstnid:
                    supersrcnid,superdstnid = superdstnid,supersrcnid
                if (supersrcnid,superdstnid) not in superEdgeConstraints:
                    superEdgeConstraints[(supersrcnid,superdstnid)] = self.edge_constraints[etup]
                    superEdgeObjectives[(supersrcnid,superdstnid)] = self.edge_objectives[etup]
                else:
                    superEdgeConstraints[(supersrcnid,superdstnid)] += self.edge_constraints[etup]
                    superEdgeObjectives[(supersrcnid,superdstnid)] += self.edge_objectives[etup]
            else:   #the edge is a part of some supernode
                if supersrcnid not in superNodeConstraints:
                    superNodeConstraints[supersrcnid] = self.edge_constraints[etup]
                    superNodeObjectives[supersrcnid] = self.edge_objectives[etup]
                else:
                    superNodeConstraints[supersrcnid] += self.edge_constraints[etup]
                    superNodeObjectives[supersrcnid] += self.edge_objectives[etup]
        for ni in self.Nodes():
            nid = ni.GetId()
            supernid = nidToSuperidMap[nid]
            value = None
            for (varID, varName, var, offset) in self.node_variables[nid]:
                if var.size[0] == 1:
                    val = numpy.array([var.value])
                else:
                    val = numpy.array(var.value).reshape(-1,)
                if not value:
                    value = val
                else:
                    value = numpy.concatenate((value, val))
            if supernid not in superNodeConstraints:
                superNodeObjectives[supernid] = self.node_objectives[nid]
                superNodeConstraints[supernid] = self.node_constraints[nid]
            else:
                superNodeObjectives[supernid] += self.node_objectives[nid]
                superNodeConstraints[supernid] += self.node_constraints[nid]
            for ( varId, varName, var, offset) in self.node_variables[nid]:
                superVarName = varName+str(varId)
                varToSuperVarMap[(nid,varName)] = (supernid,superVarName)
                if supernid not in superNodeVariables:
                    superNodeVariables[supernid] = [(varId, superVarName, var, offset)]
                    superNodeValues[supernid] = value
                else:
                    superNodeOffset = sum([superNodeVariables[supernid][k][2].size[0]* \
                                           superNodeVariables[supernid][k][2].size[1]\
                                           for k in range(builtins.len(superNodeVariables[supernid])) ])
                    superNodeVariables[supernid] += [(varId, superVarName, var, superNodeOffset)]
                    superNodeValues[supernid] = numpy.concatenate((superNodeValues[supernid],value))
                
        #add all supernodes to the supergraph
        for supernid in superNodeConstraints:
            supergraph.AddNode(supernid, superNodeObjectives[supernid], \
                               superNodeConstraints[supernid])
            supergraph.node_variables[supernid] = superNodeVariables[supernid]
            supergraph.node_values[supernid] = superNodeValues[supernid]
                        
        #add all superedges to the supergraph    
        for superei in superEdgeConstraints:
            superSrcId,superDstId = superei
            supergraph.AddEdge(superSrcId, superDstId, None,\
                               superEdgeObjectives[superei],\
                                superEdgeConstraints[superei])
                 
        #call solver for this supergraph
        if UseADMM and supergraph.GetEdges() != 0:
            supergraph.__SolveADMM(numProcessors, rho_param, maxIters, eps_abs, eps_rel, verbose)
        else:
            supergraph.Solve(M, False, numProcessors, rho_param, maxIters, eps_abs, eps_rel, verbose,
                             UseClustering=False)
        
        self.status = supergraph.status
        self.value = supergraph.value
        for ni in self.Nodes():
            nid = ni.GetId()
            snid = nidToSuperidMap[nid]
            self.node_values[nid] = []
            for ( varId, varName, var, offset) in self.node_variables[nid]:
                superVarName = varToSuperVarMap[(nid,varName)]
                self.node_values[nid] = numpy.concatenate((self.node_values[nid],\
                                                          supergraph.GetNodeValue(snid, superVarName[1])))
                    
    # Implementation of distributed ADMM
    # Uses a global value of rho_param for rho
    # Will run for a maximum of maxIters iterations
    def __SolveADMM(self, numProcessors, rho_param, maxIters, eps_abs, eps_rel,
                    verbose):
        global node_vals, edge_z_vals, edge_u_vals, rho
        global getValue, rho_update_func

        if numProcessors <= 0:
            num_processors = multiprocessing.cpu_count()
        else:
            num_processors = numProcessors
        rho = rho_param
        if verbose:
            print('Distributed ADMM (%d processors)' % num_processors)

        # Organize information for each node in helper node_info structure
        node_info = {}
        # Keeps track of the current offset necessary into the shared node
        # values Array
        length = 0
        for ni in self.Nodes():
            nid = ni.GetId()
            deg = ni.GetDeg()
            obj = self.node_objectives[nid]
            variables = self.node_variables[nid]
            con = self.node_constraints[nid]
            neighbors = [ni.GetNbrNId(j) for j in range(deg)]
            # Node's constraints include those imposed by edges
            for neighborId in neighbors:
                etup = self.__GetEdgeTup(nid, neighborId)
                econ = self.edge_constraints[etup]
                con += econ
            # Calculate sum of dimensions of all Variables for this node
            size = sum([var.size[0] for (varID, varName, var, offset) in variables])
            # Nearly complete information package for this node
            node_info[nid] = (nid, obj, variables, con, length, size, deg,\
                neighbors)
            length += size
        node_vals = multiprocessing.Array('d', [0.0] * length)
        x_length = length

        # Organize information for each node in final edge_list structure and
        # also helper edge_info structure
        edge_list = []
        edge_info = {}
        # Keeps track of the current offset necessary into the shared edge
        # values Arrays
        length = 0
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            obj = self.edge_objectives[etup]
            con = self.edge_constraints[etup]
            con += self.node_constraints[etup[0]] +\
                self.node_constraints[etup[1]]
            # Get information for each endpoint node
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            ind_zij = length
            ind_uij = length
            length += info_i[X_LEN]
            ind_zji = length
            ind_uji = length
            length += info_j[X_LEN]
            # Information package for this edge
            tup = (etup, obj, con,\
                info_i[X_VARS], info_i[X_LEN], info_i[X_IND], ind_zij, ind_uij,\
                info_j[X_VARS], info_j[X_LEN], info_j[X_IND], ind_zji, ind_uji)
            edge_list.append(tup)
            edge_info[etup] = tup
        edge_z_vals = multiprocessing.Array('d', [0.0] * length)
        edge_u_vals = multiprocessing.Array('d', [0.0] * length)
        z_length = length

        # Populate sparse matrix A.
        # A has dimensions (p, n), where p is the length of the stacked vector
        # of node variables, and n is the length of the stacked z vector of
        # edge variables.
        # Each row of A has one 1. There is a 1 at (i,j) if z_i = x_j.
        A = lil_matrix((z_length, x_length), dtype=numpy.int8)
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            info_edge = edge_info[etup]
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            for offset in range(info_i[X_LEN]):
                row = info_edge[Z_ZIJIND] + offset
                col = info_i[X_IND] + offset
                A[row, col] = 1
            for offset in range(info_j[X_LEN]):
                row = info_edge[Z_ZJIIND] + offset
                col = info_j[X_IND] + offset
                A[row, col] = 1
        A_tr = A.transpose()

        # Create final node_list structure by adding on information for
        # node neighbors
        node_list = []
        for nid, info in node_info.iteritems():
            entry = [nid, info[X_OBJ], info[X_VARS], info[X_CON], info[X_IND],\
                info[X_LEN], info[X_DEG]]
            # Append information about z- and u-value indices for each
            # node neighbor
            for i in range(info[X_DEG]):
                neighborId = info[X_NEIGHBORS][i]
                indices = (Z_ZIJIND, Z_UIJIND) if nid < neighborId else\
                    (Z_ZJIIND, Z_UJIIND)
                einfo = edge_info[self.__GetEdgeTup(nid, neighborId)]
                entry.append(einfo[indices[0]])
                entry.append(einfo[indices[1]])
            node_list.append(entry)

        pool = multiprocessing.Pool(num_processors)
        num_iterations = 0
        z_old = getValue(edge_z_vals, 0, z_length)
        # Proceed until convergence criteria are achieved or the maximum
        # number of iterations has passed
        while num_iterations <= maxIters:
            # Check convergence criteria
            if num_iterations != 0:
                x = getValue(node_vals, 0, x_length)
                z = getValue(edge_z_vals, 0, z_length)
                u = getValue(edge_u_vals, 0, z_length)
                # Determine if algorithm should stop. Retrieve primal and dual
                # residuals and thresholds
                stop, res_pri, e_pri, res_dual, e_dual =\
                    self.__CheckConvergence(A, A_tr, x, z, z_old, u, rho,\
                                            x_length, z_length,
                                            eps_abs, eps_rel, verbose)
                if stop: break
                z_old = z
                # Update rho and scale u-values
                rho_new = rho_update_func(rho, res_pri, e_pri, res_dual, e_dual)
                scale = float(rho) / rho_new
                edge_u_vals[:] = [i * scale for i in edge_u_vals]
                rho = rho_new
            num_iterations += 1

            if verbose:
                # Debugging information prints current iteration #
                print('Iteration %d' % num_iterations)
            pool.map(ADMM_x, node_list)
            pool.map(ADMM_z, edge_list)
            pool.map(ADMM_u, edge_list)
        pool.close()
        pool.join()

        # Insert into hash to support GetNodeValue()
        for entry in node_list:
            nid = entry[X_NID]
            index = entry[X_IND]
            size = entry[X_LEN]
            self.node_values[nid] = getValue(node_vals, index, size)
        # Set TGraphVX status and value to match CVXPY
        if num_iterations <= maxIters:
            self.status = 'Optimal'
        else:
            self.status = 'Incomplete: max iterations reached'
        self.value = self.GetTotalProblemValue()

    # Iterate through all variables and update values.
    # Sum all objective values over all nodes and edges.
    def GetTotalProblemValue(self):
        global getValue
        result = 0.0
        for ni in self.Nodes():
            nid = ni.GetId()
            for (varID, varName, var, offset) in self.node_variables[nid]:
                var.value = self.GetNodeValue(nid, varName)
        for ni in self.Nodes():
            result += self.node_objectives[ni.GetId()].value
        for ei in self.Edges():
            etup = self.__GetEdgeTup(ei.GetSrcNId(), ei.GetDstNId())
            result += self.edge_objectives[etup].value
        return result

    # Returns True if convergence criteria have been satisfied
    # eps_abs = eps_rel = 0.01
    # r = Ax - z
    # s = rho * (A^T)(z - z_old)
    # e_pri = sqrt(p) * e_abs + e_rel * max(||Ax||, ||z||)
    # e_dual = sqrt(n) * e_abs + e_rel * ||rho * (A^T)u||
    # Should stop if (||r|| <= e_pri) and (||s|| <= e_dual)
    # Returns (boolean shouldStop, primal residual value, primal threshold,
    #          dual residual value, dual threshold)
    def __CheckConvergence(self, A, A_tr, x, z, z_old, u, rho, p, n,
                           e_abs, e_rel, verbose):
        norm = numpy.linalg.norm
        Ax = A.dot(x)
        r = Ax - z
        s = rho * A_tr.dot(z - z_old)
        # Primal and dual thresholds. Add .0001 to prevent the case of 0.
        e_pri = math.sqrt(p) * e_abs + e_rel * max(norm(Ax), norm(z)) + .0001
        e_dual = math.sqrt(n) * e_abs + e_rel * norm(rho * A_tr.dot(u)) + .0001
        # Primal and dual residuals
        res_pri = norm(r)
        res_dual = norm(s)
        if verbose:
            # Debugging information to print(convergence criteria values)
            print('  r:', res_pri)
            print('  e_pri:', e_pri)
            print('  s:', res_dual)
            print('  e_dual:', e_dual)
        stop = (res_pri <= e_pri) and (res_dual <= e_dual)
        return (stop, res_pri, e_pri, res_dual, e_dual)

    # API to get node Variable value after solving with ADMM.
    def GetNodeValue(self, NId, Name):
        self.__VerifyNId(NId)
        for (varID, varName, var, offset) in self.node_variables[NId]:
            if varName == Name:
                offset = offset
                value = self.node_values[NId]
                return value[offset:(offset + var.size[0])]
        return None

    # Prints value of all node variables to console or file, if given
    def PrintSolution(self, Filename=None):
        numpy.set_printoptions(linewidth=numpy.inf)
        out = sys.stdout if (Filename == None) else open(Filename, 'w+')

        out.write('Status: %s\n' % self.status)
        out.write('Total Objective: %f\n' % self.value)
        for ni in self.Nodes():
            nid = ni.GetId()
            s = 'Node %d:\n' % nid
            out.write(s)
            for (varID, varName, var, offset) in self.node_variables[nid]:
                val = numpy.transpose(self.GetNodeValue(nid, varName))
                s = '  %s %s\n' % (varName, str(val))
                out.write(s)

    # Helper method to verify existence of an NId.
    def __VerifyNId(self, NId):
        if not TUNGraph.IsNode(self, NId):
            raise Exception('Node %d does not exist.' % NId)

    # Helper method to determine if
    def __UpdateAllVariables(self, NId, Objective):
        if NId in self.node_objectives:
            # First, remove the Variables from the old Objective.
            old_obj = self.node_objectives[NId]
            self.all_variables = self.all_variables - set(old_obj.variables())
        # Check that the Variables of the new Objective are not currently
        # in other Objectives.
        new_variables = set(Objective.variables())
        if builtins.len(self.all_variables.intersection(new_variables)) != 0:
            raise Exception('Objective at NId %d shares a variable.' % NId)
        self.all_variables = self.all_variables | new_variables

    # Helper method to get CVXPY Variables out of a CVXPY Objective
    def __ExtractVariableList(self, Objective):
        l = [(var.name(), var) for var in Objective.variables()]
        # Sort in ascending order by name
        l.sort(key=lambda t: t[0])
        l2 = []
        offset = 0
        for (varName, var) in l:
            # Add tuples of the form (id, name, object, offset)
            l2.append((var.id, varName, var, offset))
            offset += var.size[0]
        return l2

    # Adds a Node to the TUNGraph and stores the corresponding CVX information.
    def AddNode(self, NId, Objective=__default_objective,\
            Constraints=__default_constraints):
        self.__UpdateAllVariables(NId, Objective)
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = self.__ExtractVariableList(Objective)
        self.node_constraints[NId] = Constraints
        return TUNGraph.AddNode(self, NId)

    def SetNodeObjective(self, NId, Objective):
        self.__VerifyNId(NId)
        self.__UpdateAllVariables(NId, Objective)
        self.node_objectives[NId] = Objective
        self.node_variables[NId] = self.__ExtractVariableList(Objective)

    def GetNodeObjective(self, NId):
        self.__VerifyNId(NId)
        return self.node_objectives[NId]

    def SetNodeConstraints(self, NId, Constraints):
        self.__VerifyNId(NId)
        self.node_constraints[NId] = Constraints

    def GetNodeConstraints(self, NId):
        self.__VerifyNId(NId)
        return self.node_constraints[NId]

    # Helper method to get a tuple representing an edge. The smaller NId
    # goes first.
    def __GetEdgeTup(self, NId1, NId2):
        return (NId1, NId2) if NId1 < NId2 else (NId2, NId1)

    # Helper method to verify existence of an edge.
    def __VerifyEdgeTup(self, ETup):
        if not TUNGraph.IsEdge(self, ETup[0], ETup[1]):
            raise Exception('Edge {%d,%d} does not exist.' % ETup)

    # Adds an Edge to the TUNGraph and stores the corresponding CVX information.
    # obj_func is a function which accepts two arguments, a dictionary of
    #     variables for the source and destination nodes
    #     { string varName : CVXPY Variable }
    # obj_func should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective and will use
    #     the default constraints.
    # If obj_func is None, then will use Objective and Constraints, which are
    #     parameters currently set to defaults.
    def AddEdge(self, SrcNId, DstNId, ObjectiveFunc=None,
            Objective=__default_objective, Constraints=__default_constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        if ObjectiveFunc != None:
            src_vars = self.GetNodeVariables(SrcNId)
            dst_vars = self.GetNodeVariables(DstNId)
            ret = ObjectiveFunc(src_vars, dst_vars)
            if type(ret) is tuple:
                # Tuple = assume we have (objective, constraints)
                self.edge_objectives[ETup] = ret[0]
                self.edge_constraints[ETup] = ret[1]
            else:
                # Singleton object = assume it is the objective
                self.edge_objectives[ETup] = ret
                self.edge_constraints[ETup] = self.__default_constraints
        else:
            self.edge_objectives[ETup] = Objective
            self.edge_constraints[ETup] = Constraints
        return TUNGraph.AddEdge(self, SrcNId, DstNId)

    def SetEdgeObjective(self, SrcNId, DstNId, Objective):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        self.edge_objectives[ETup] = Objective

    def GetEdgeObjective(self, SrcNId, DstNId):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        return self.edge_objectives[ETup]

    def SetEdgeConstraints(self, SrcNId, DstNId, Constraints):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        self.edge_constraints[ETup] = Constraints

    def GetEdgeConstraints(self, SrcNId, DstNId):
        ETup = self.__GetEdgeTup(SrcNId, DstNId)
        self.__VerifyEdgeTup(ETup)
        return self.edge_constraints[ETup]


    # Returns a dictionary of all variables corresponding to a node.
    # { string name : CVXPY Variable }
    # This can be used in place of bulk loading functions to recover necessary
    # Variables for an edge.
    def GetNodeVariables(self, NId):
        self.__VerifyNId(NId)
        d = {}
        for (varID, varName, var, offset) in self.node_variables[NId]:
            d[varName] = var
        return d

    # Bulk loading for nodes
    # ObjFunc is a function which accepts one argument, an array of strings
    #     parsed from the given CSV filename
    # ObjFunc should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective
    # Optional parameter NodeIDs allows the user to pass in a list specifying,
    # in order, the node IDs that correspond to successive rows
    # If NodeIDs is None, then the file must have a column denoting the
    # node ID for each row. The index of this column (0-indexed) is IdCol.
    # If NodeIDs and IdCol are both None, then will iterate over all Nodes, in
    # order, as long as the file lasts
    def AddNodeObjectives(self, Filename, ObjFunc, NodeIDs=None, IdCol=None):
        infile = open(Filename, 'r')
        if NodeIDs == None and IdCol == None:
            stop = False
            for ni in self.Nodes():
                nid = ni.GetId()
                while True:
                    line = infile.readline()
                    if line == '': stop = True
                    if not line.startswith('#'): break
                if stop: break
                data = [x.strip() for x in line.split(',')]
                ret = ObjFunc(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(nid, ret[0])
                    self.SetNodeConstraints(nid, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(nid, ret)
        if NodeIDs == None:
            for line in infile:
                if line.startswith('#'): continue
                data = [x.strip() for x in line.split(',')]
                ret = ObjFunc(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(int(data[IdCol]), ret[0])
                    self.SetNodeConstraints(int(data[IdCol]), ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(int(data[IdCol]), ret)
        else:
            for nid in NodeIDs:
                while True:
                    line = infile.readline()
                    if line == '':
                        raise Exception('File %s is too short.' % Filename)
                    if not line.startswith('#'): break
                data = [x.strip() for x in line.split(',')]
                ret = ObjFunc(data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetNodeObjective(nid, ret[0])
                    self.SetNodeConstraints(nid, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetNodeObjective(nid, ret)
        infile.close()

    # Bulk loading for edges
    # If Filename is None:
    # ObjFunc is a function which accepts three arguments, a dictionary of
    #     variables for the source and destination nodes, and an unused param
    #     { string varName : CVXPY Variable } x2, None
    # ObjFunc should return a tuple of (objective, constraints), although
    #     it will assume a singleton object will be an objective
    # If Filename exists:
    # ObjFunc is the same, except the third param will be be an array of
    #     strings parsed from the given CSV filename
    # Optional parameter EdgeIDs allows the user to pass in a list specifying,
    # in order, the EdgeIDs that correspond to successive rows. An edgeID is
    # a tuple of (srcID, dstID).
    # If EdgeIDs is None, then the file may have columns denoting the srcID and
    # dstID for each row. The indices of these columns are 0-indexed.
    # If EdgeIDs and id columns are None, then will iterate through all edges
    # in order, as long as the file lasts.
    def AddEdgeObjectives(self, ObjFunc, Filename=None, EdgeIDs=None,\
            SrcIdCol=None, DstIdCol=None):
        if Filename == None:
            for ei in self.Edges():
                src_id = ei.GetSrcNId()
                src_vars = self.GetNodeVariables(src_id)
                dst_id = ei.GetDstNId()
                dst_vars = self.GetNodeVariables(dst_id)
                ret = ObjFunc(src_vars, dst_vars, None)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(src_id, dst_id, ret[0])
                    self.SetEdgeConstraints(src_id, dst_id, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(src_id, dst_id, ret)
            return
        infile = open(Filename, 'r')
        if EdgeIDs == None and (SrcIdCol == None or DstIdCol == None):
            stop = False
            for ei in self.Edges():
                src_id = ei.GetSrcNId()
                src_vars = self.GetNodeVariables(src_id)
                dst_id = ei.GetDstNId()
                dst_vars = self.GetNodeVariables(dst_id)
                while True:
                    line = infile.readline()
                    if line == '': stop = True
                    if not line.startswith('#'): break
                if stop: break
                data = [x.strip() for x in line.split(',')]
                ret = ObjFunc(src_vars, dst_vars, data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(src_id, dst_id, ret[0])
                    self.SetEdgeConstraints(src_id, dst_id, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(src_id, dst_id, ret)
        if EdgeIDs == None:
            for line in infile:
                if line.startswith('#'): continue
                data = [x.strip() for x in line.split(',')]
                src_id = int(data[SrcIdCol])
                dst_id = int(data[DstIdCol])
                src_vars = self.GetNodeVariables(src_id)
                dst_vars = self.GetNodeVariables(dst_id)
                ret = ObjFunc(src_vars, dst_vars, data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(src_id, dst_id, ret[0])
                    self.SetEdgeConstraints(src_id, dst_id, ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(src_id, dst_id, ret)
        else:
            for edgeID in EdgeIDs:
                etup = self.__GetEdgeTup(edgeID[0], edgeID[1])
                while True:
                    line = infile.readline()
                    if line == '':
                        raise Exception('File %s is too short.' % Filename)
                    if not line.startswith('#'): break
                data = [x.strip() for x in line.split(',')]
                src_vars = self.GetNodeVariables(etup[0])
                dst_vars = self.GetNodeVariables(etup[1])
                ret = ObjFunc(src_vars, dst_vars, data)
                if type(ret) is tuple:
                    # Tuple = assume we have (objective, constraints)
                    self.SetEdgeObjective(etup[0], etup[1], ret[0])
                    self.SetEdgeConstraints(etup[0], etup[1], ret[1])
                else:
                    # Singleton object = assume it is the objective
                    self.SetEdgeObjective(etup[0], etup[1], ret)
        infile.close()

    """return clusters of nodes of the original graph.Each cluster corresponds to 
    a supernode in the supergraph"""
    def __ClusterGraph(self,clusterSize):
        #obtain a random shuffle of the nodes
        nidArray = [ni.GetId() for ni in self.Nodes()]
        numpy.random.shuffle(nidArray)
        visitedNode = {}
        for nid in nidArray:
            visitedNode[nid] = False
        superNodes = []
        superNode,superNodeSize = [],0
        for nid in nidArray:
            if not visitedNode[nid]:
                oddLevel, evenLevel, isOdd = [],[],True
                oddLevel.append(nid)
                visitedNode[nid] = True
                #do a level order traversal and add nodes to the superNode until the 
                #size of the supernode variables gets larger than clusterSize
                while True:
                    if isOdd:
                        if builtins.len(oddLevel) > 0:
                            while builtins.len(oddLevel) > 0:
                                topId = oddLevel.pop(0)
                                node = TUNGraph.GetNI(self,topId)
                                varSize = sum([variable[2].size[0]* \
                                               variable[2].size[1]\
                                               for variable in self.node_variables[topId]])
                                if varSize + superNodeSize <= clusterSize:
                                    superNode.append(topId)
                                    superNodeSize = varSize + superNodeSize
                                else:
                                    if builtins.len(superNode) > 0:
                                        superNodes.append(superNode)
                                    superNodeSize = varSize
                                    superNode = [topId]
                                neighbors = [node.GetNbrNId(j) \
                                             for j in range(node.GetDeg())]
                                for nbrId in neighbors:
                                    if not visitedNode[nbrId]:
                                        evenLevel.append(nbrId)
                                        visitedNode[nbrId] = True
                            isOdd = False
                            #sort the nodes according to their variable size
                            if builtins.len(evenLevel) > 0:
                                evenLevel.sort(key=lambda nid : sum([variable[2].size[0]* \
                                               variable[2].size[1] for variable \
                                               in self.node_variables[nid]]))
                        else:
                            break
                    else:
                        if builtins.len(evenLevel) > 0:
                            while builtins.len(evenLevel) > 0:
                                topId = evenLevel.pop(0)
                                node = TUNGraph.GetNI(self,topId)
                                varSize = sum([variable[2].size[0]* \
                                               variable[2].size[1]\
                                               for variable in self.node_variables[topId]])
                                if varSize + superNodeSize <= clusterSize:
                                    superNode.append(topId)
                                    superNodeSize = varSize + superNodeSize
                                else:
                                    if builtins.len(superNode) > 0:
                                        superNodes.append(superNode)
                                    superNodeSize = varSize
                                    superNode = [topId]
                                neighbors = [node.GetNbrNId(j) \
                                             for j in range(node.GetDeg())]
                                for nbrId in neighbors:
                                    if not visitedNode[nbrId]:
                                        oddLevel.append(nbrId)
                                        visitedNode[nbrId] = True
                            isOdd = True
                            #sort the nodes according to their variable size
                            if builtins.len(oddLevel) > 0:
                                oddLevel.sort(key=lambda nid : sum([variable[2].size[0]* \
                                               variable[2].size[1] for variable \
                                               in self.node_variables[nid]]))
                        else:
                            break
        if superNode not in superNodes:
            superNodes.append(superNode)
        return superNodes
## ADMM Global Variables and Functions ##

# By default, the objective function is Minimize().
__default_m_func = Minimize
m_func = __default_m_func

# By default, rho is 1.0. Default rho update is identity function and does not
# depend on primal or dual residuals or thresholds.
__default_rho = 1.0
__default_rho_update_func = lambda rho, res_p, thr_p, res_d, thr_d: rho
rho = __default_rho
# Rho update function takes 5 parameters
# - Old value of rho
# - Primal residual and threshold
# - Dual residual and threshold
rho_update_func = __default_rho_update_func

def SetRho(Rho=None):
    global rho
    rho = Rho if Rho else __default_rho

# Rho update function should take one parameter: old_rho
# Returns new_rho
# This function will be called at the end of every iteration
def SetRhoUpdateFunc(Func=None):
    global rho_update_func
    rho_update_func = Func if Func else __default_rho_update_func

# Tuple of indices to identify the information package for each node. Actual
# length of specific package (list) may vary depending on node degree.
# X_NID: Node ID
# X_OBJ: CVXPY Objective
# X_VARS: CVXPY Variables (entry from node_variables structure)
# X_CON: CVXPY Constraints
# X_IND: Starting index into shared node_vals Array
# X_LEN: Total length (sum of dimensions) of all variables
# X_DEG: Number of neighbors
# X_NEIGHBORS: Placeholder for information about each neighbors
#   Information for each neighbor is two entries, appended in order.
#   Starting index of the corresponding z-value in edge_z_vals. Then for u.
(X_NID, X_OBJ, X_VARS, X_CON, X_IND, X_LEN, X_DEG, X_NEIGHBORS) = range(8)

# Tuple of indices to identify the information package for each edge.
# Z_EID: Edge ID / tuple
# Z_OBJ: CVXPY Objective
# Z_CON: CVXPY Constraints
# Z_[IJ]VARS: CVXPY Variables for Node [ij] (entry from node_variables)
# Z_[IJ]LEN: Total length (sum of dimensions) of all variables for Node [ij]
# Z_X[IJ]IND: Starting index into shared node_vals Array for Node [ij]
# Z_Z[IJ|JI]IND: Starting index into shared edge_z_vals Array for edge [ij|ji]
# Z_U[IJ|JI]IND: Starting index into shared edge_u_vals Array for edge [ij|ji]
(Z_EID, Z_OBJ, Z_CON, Z_IVARS, Z_ILEN, Z_XIIND, Z_ZIJIND, Z_UIJIND,\
    Z_JVARS, Z_JLEN, Z_XJIND, Z_ZJIIND, Z_UJIIND) = range(13)

# Contain all x, z, and u values for each node and/or edge in ADMM. Use the
# given starting index and length with getValue() to get individual node values
node_vals = None
edge_z_vals = None
edge_u_vals = None

# Extract a numpy array value from a shared Array.
# Give shared array, starting index, and total length.
def getValue(arr, index, length):
    return numpy.array(arr[index:(index + length)])

# Write value of numpy array nparr (with given length) to a shared Array at
# the given starting index.
def writeValue(sharedarr, index, nparr, length):
    if length == 1:
        nparr = [nparr]
    sharedarr[index:(index + length)] = nparr

# Write the values for all of the Variables involved in a given Objective to
# the given shared Array.
# variables should be an entry from the node_values structure.
def writeObjective(sharedarr, index, objective, variables):
    for v in objective.variables():
        vID = v.id
        value = v.value
        # Find the tuple in variables with the same ID. Take the offset.
        # If no tuple exists, then silently skip.
        for (varID, varName, var, offset) in variables:
            if varID == vID:
                writeValue(sharedarr, index + offset, value, var.size[0])
                break
# Proximal operators
def Prox_logdet(S, A, eta):
    global rho
    d, q = numpy.linalg.eigh(eta*A-S)
    q = numpy.matrix(q)
    X_var = ( 1/(2*eta) )*q*( numpy.diag(d + numpy.sqrt(numpy.square(d) + (4*eta)*numpy.ones(d.shape))) )*q.T
    x_var = X_var[numpy.triu_indices(S.shape[1])] # extract upper triangular part as update variable      
#        print('x_update = ',x_var)
    return numpy.matrix(x_var).T
    
def Prox_lasso(a_ij, a_ji, eta, NID_diff):   
    z_ij = numpy.copy(a_ij)
    z_ji = numpy.copy(a_ji)
    
    k = 0
    ind = range(a_ij.shape[0])
    n = int((-1  + numpy.sqrt(1+ 8*a_ij.shape[0]))/2)  
    for i in range(n,0,-1):
        ind.remove(k)
        k = k + i

    if (NID_diff > 1):
        z_ij[ind] = Prox_onenorm(a_ij[ind], eta)
    else:
        z_ji[ind] = Prox_onenorm(a_ji[ind], eta)
#    if (NID_diff > 1): # for lasso penality between node i with logdet and dummynode j:
##        print('we are in lasso penalty edge, alpha = ', entry[1].args[0].value)
#        ind = (a_ij > eta)
#        z_ij[ind] = a_ij[ind]-eta
#    else:
#        ind = (a_ij < -eta)
#        z_ij[ind] = a_ij[ind] + eta
        
    return z_ij.T, z_ji.T

def Prox_onenorm(A, eta):
    Z = numpy.zeros(A.shape)
    ind = (A > eta)
    Z[ind] = A[ind] - eta
    ind = (A < - eta)
    Z[ind] = A[ind] + eta
    return Z
    
def Prox_twonorm(A, eta):
    col_norms = numpy.linalg.norm(A, axis = 0) 
    Z = numpy.dot(A, numpy.diag((numpy.ones(A.shape[0]) - eta/col_norms)*(col_norms > eta))) 
    return Z 


def Prox_penalty(a_ij, a_ji, eta, index_penalty):
    n = int((-1  + numpy.sqrt(1+ 8*a_ij.shape[0]))/2)      
    z_ij = (a_ij+a_ji)/2
    z_ji = (a_ij+a_ji)/2
    alpha = 2
    d = 0
    # For onenorm, infnorm, and laplacian penalty, the solution is computed elementwise
    # Otherwise, the full matrix should be formed in order to compute the solution.
    if index_penalty == 1:
        e = Prox_onenorm(a_ij - a_ji + d, alpha*eta) -d
    elif index_penalty == 3:        
        e = (a_ij - a_ji + d)/(1 + 2*alpha*eta) -d
    else:
        A_ij = upper2Full(a_ij)
        A_ji = upper2Full(a_ji)
#        print(A_2 - A_ji)
        if index_penalty == 2:  
            e = Prox_twonorm(A_ij - A_ji + d, alpha*eta) -d  
        #elif index_penalty == 4:
        #    e = Prox_infnorm(A_ij - A_ji + d, alpha*eta) -d
        else:
            MaxIter = 100   
            eps = 1e-3
            [Z_ij, Z_ji] = Prox_node_penalty(A_ij, A_ji, eta, MaxIter, eps)
            z_ij = (numpy.squeeze(numpy.asarray(Z_ij[numpy.triu_indices(n)])))
            z_ji = (numpy.squeeze(numpy.asarray(Z_ji[numpy.triu_indices(n)])))
            return z_ij, z_ji
        e = e[numpy.triu_indices(n)]
    e = e/alpha
    z_ij = z_ij + e
    z_ji = z_ji - e
    return z_ij, z_ji
  
def Prox_node_penalty(A_ij, A_ji, beta, MaxIter, eps):
    global rho
    n = A_ij.shape[0]
    I = numpy.identity(n)  
    U = numpy.ones([n,n])/n
    U1 = numpy.ones([n,n])/n
    U2 = numpy.ones([n,n])/n
    theta_1 = numpy.copy(U)
    theta_2 = numpy.copy(U)
    V = numpy.copy(U)
    W = numpy.copy(U)
    
    for k in range(MaxIter):
        A = ((theta_1 - theta_2 - W - U1) + (W.T - U2))/2
        eta = beta/(2*rho)
        V = Prox_twonorm(A, eta)
    
        eta = (rho/2)/rho
        C = numpy.concatenate((I,-I, I), axis = 1)        
        C = numpy.matrix(C)
        A = numpy.concatenate([(V+U2).T, A_ij, A_ji], axis = 0)
        D = V + U1

        Z = numpy.linalg.solve(C.T*C + eta*numpy.identity(3*n), - C.T*D + eta* A)
        W = Z[:n,:]
        theta_1 = Z[n:2*n,:]
        theta_2 = Z[2*n:,:]    
        
        deltaU1 = ((V + W) - (theta_1 - theta_2)) 
        deltaU2 = (V - W.T)
        if numpy.linalg.norm(deltaU1,'fro')<eps and numpy.linalg.norm(deltaU1,'fro')<eps:
            U1 = U1 + deltaU1
            U2 = U2 + deltaU2
#            print('iteration number is', k)
            break
        U1 = U1 + deltaU1
        U2 = U2 + deltaU2
    return theta_1,theta_2
    
    

def upper2Full(a):
    n = int((-1  + numpy.sqrt(1+ 8*a.shape[0]))/2)  
    A = numpy.zeros([n,n])
    A[numpy.triu_indices(n)] = a 
    temp = A.diagonal()
    A = (A + A.T) - numpy.diag(temp)             
    return A   
    
# x-update for ADMM for one node
def ADMM_x(entry):
    global rho
    variables = entry[X_VARS]
#    norms = 0
    
    #-----------------------Proximal operator ---------------------------
    x_update = [] # proximal update for the variable x
    if(builtins.len(entry[1].args) > 1 ):
#        print('we are in logdet + trace node')
        cvxpyMat = entry[1].args[1].args[0].args[0]
        numpymat = cvxpyMat.value
        n_t=1 # Assume number of samples is 1 at each node, need to be alterned alter
        # Iterate through all neighbors of the node
        mat_shape = (int(numpymat.shape[1] * ( numpymat.shape[1]+1 )/2.0),)
        a = numpy.zeros(mat_shape) 
#        print('degree = ', entry[X_DEG])
        for i in range(entry[X_DEG]):  # entry[X_DEG] = 3 if the node is neither first and the last one    
            z_index = X_NEIGHBORS + (2 * i)
            u_index = z_index + 1
            zi = entry[z_index]
            ui = entry[u_index]
            
            # Add norm for Variables corresponding to the node
            for (varID, varName, var, offset) in variables:
                z = getValue(edge_z_vals, zi + offset, var.size[0])
                u = getValue(edge_u_vals, ui + offset, var.size[0])
                a += (z-u) 
        A = upper2Full(a)
        A =  A/entry[X_DEG]
#        d, q = numpy.linalg.eigh((rho*A/n_t)-numpymat)
#        q = numpy.matrix(q)
        eta = entry[X_DEG]*rho/n_t
#        X = ( 1/(2*eta) )*q*( numpy.diag(d + numpy.sqrt(numpy.square(d) + (4*eta)*numpy.ones(d.shape))) )*q.T
#        x_var = X[numpy.triu_indices(numpymat.shape[1])] # extract upper triangular part as update variable      
#        print('x_update = ',x_var)
#        solution = numpy.matrix(x_var).T
        x_update = Prox_logdet(numpymat, A, eta)
        solution = numpy.array(x_update).T.reshape(-1)
        writeValue(node_vals, entry[X_IND] + variables[0][3], solution, variables[0][2].size[0]) 
    else:
#        print('we are in the dummy node')
        x_update = [] # no variable to update for dummy node
    #-----------------------Proximal operator ---------------------------
#    print('end of proximal operator')

#
#    #----------------------- Use CVXPY  -----------------------------
#    # Iterate through all neighbors of the node
#    for i in range(entry[X_DEG]):
#        z_index = X_NEIGHBORS + (2 * i)
#        u_index = z_index + 1
#        zi = entry[z_index]
#        ui = entry[u_index]
#        # Add norm for Variables corresponding to the node
#        for (varID, varName, var, offset) in variables:
#            z = getValue(edge_z_vals, zi + offset, var.size[0])
#            u = getValue(edge_u_vals, ui + offset, var.size[0])
#            norms += square(norm(var - z + u))
#
#    objective = entry[X_OBJ] + (rho / 2) * norms
#    objective = m_func(objective)
#    constraints = entry[X_CON]
#    problem = Problem(objective, constraints)
#    try:
#        problem.solve()
#    except SolverError:
#        problem.solve(solver=SCS)
#    if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
#        print("ECOS error: using SCS for x update")
#        problem.solve(solver=SCS)
#
#    #----------------------- Use CVXPY  -----------------------------
##    print('end of cvxpy_x')
#
##    for v in objective.variables():
##        value = v.value
##        print('variable z_ij or z_ji = ', value)
##        
##    print('x gotten by proximal operator', x_var)
#
#    # Write back result of x-update
#    writeObjective(node_vals, entry[X_IND], objective, variables)
    return None

# z-update for ADMM for one edge
def ADMM_z(entry, index_penalty = 1):
    global rho
    
    #Select this parameter to determine which edge penalty to use:
    #1: L1-norm, 2: L2-norm, 3: Laplacian, 4: L-inf norm, 5: Perturbed-node
    index_penalty = 2
    
    #-----------------------Proximal operator ---------------------------
    if index_penalty != 2:
        a_ij = [] # 
        flag = 0
        variables_i = entry[Z_IVARS]
        for (varID, varName, var, offset) in variables_i:
            x_i = getValue(node_vals, entry[Z_XIIND] + offset, var.size[0])
            u_ij = getValue(edge_u_vals, entry[Z_UIJIND] + offset, var.size[0])
            if flag == 0:
                a_ij = (x_i + u_ij)
                flag = 1
            else:
                a_ij += (x_i + u_ij) 
                
        a_ji = []
        flag = 0
        variables_j = entry[Z_JVARS]
        for (varID, varName, var, offset) in variables_j:
            x_j = getValue(node_vals, entry[Z_XJIND] + offset, var.size[0])
            u_ji = getValue(edge_u_vals, entry[Z_UJIIND] + offset, var.size[0])
            if flag == 0:
                a_ji = (x_j + u_ji)
                flag = 1
            else:
                a_ji += (x_j + u_ji)
    #        
    #    z_ij = numpy.zeros(a_ij.shape)
    #    z_ji = numpy.zeros(a_ij.shape)
        NID_diff  = entry[0][1]-entry[0][0]
    #    print('entry[0] = ', entry[0], 'NID_diff = ' ,NID_diff )
    #    eta = 2*entry[1].args[0].value/rho
    #    print('alpha/beta = ', entry[1].args[0].value )
        eta = entry[1].args[0].value/rho # where entry[1].args[0].value can be alpha or bete depending on NID_diff
    
        if (numpy.abs(NID_diff) <= 1): # for psi penalty edge
    #        beta = entry[1].args[0].value
            [z_ij, z_ji] =  Prox_penalty(a_ij, a_ji, eta, index_penalty)
    #        print('we are in psi penalty edge, beta = ', entry[1].args[0].value)
        else: 
    #        print('we are in lasso penalty edge, alpha = ', entry[1].args[0].value)
            [z_ij, z_ji] = Prox_lasso(a_ij, a_ji, eta, NID_diff) 
    
        if (NID_diff >= -1):
            writeValue(edge_z_vals, entry[Z_ZIJIND] + variables_i[0][3], z_ij, variables_i[0][2].size[0])
        if (NID_diff <= 1):
            writeValue(edge_z_vals, entry[Z_ZJIIND] + variables_j[0][3], z_ji, variables_j[0][2].size[0])
#    -----------------------Proximal operator ---------------------------    
#    print('end of proximal operator')
#    
    #----------------------- Use CVXPY  -----------------------------
    else:
        objective = entry[Z_OBJ]
        constraints = entry[Z_CON]
        norms = 0    
        variables_i = entry[Z_IVARS]
    #    print('here 1')
        for (varID, varName, var, offset) in variables_i:
    #        print('var = ', var)
            x_i = getValue(node_vals, entry[Z_XIIND] + offset, var.size[0])
            u_ij = getValue(edge_u_vals, entry[Z_UIJIND] + offset, var.size[0])
            norms += square(norm(x_i - var + u_ij,'fro'))
    #        norms += square(norm(x_i - var + u_ij))
        variables_j = entry[Z_JVARS]
        
    #    print('here 2')
        for (varID, varName, var, offset) in variables_j:
            x_j = getValue(node_vals, entry[Z_XJIND] + offset, var.size[0])
            u_ji = getValue(edge_u_vals, entry[Z_UJIIND] + offset, var.size[0])
            norms += square(norm(x_j - var + u_ji,'fro'))
    #        norms += square(norm(x_j - var + u_ji))
    #    print('here 2-1')
        objective = m_func(objective + (rho / 2) * norms)
        problem = Problem(objective, constraints)
        
    #    print('here 3')
        try:
            problem.solve()
        except SolverError:
            problem.solve(solver=SCS)
        if problem.status in [INFEASIBLE_INACCURATE, UNBOUNDED_INACCURATE]:
            print("ECOS error: using SCS for z update")
            problem.solve(solver=SCS)
        
    #    jj = 0
    #    for v in objective.variables():
    #        jj = jj + 1        
    #        value = v.value
    #        print()
    #        print('jj = ', jj,'value.shape = ', value.shape        )
    
        # Write back result of z-update. Must write back for i- and j-node
        writeObjective(edge_z_vals, entry[Z_ZIJIND], objective, variables_i)
        writeObjective(edge_z_vals, entry[Z_ZJIIND], objective, variables_j)
        
    #----------------------- Use CVXPY  -----------------------------

    return None

# u-update for ADMM for one edge
def ADMM_u(entry):
    global rho
    size_i = entry[Z_ILEN]
    uij = getValue(edge_u_vals, entry[Z_UIJIND], size_i) +\
          getValue(node_vals, entry[Z_XIIND], size_i) -\
          getValue(edge_z_vals, entry[Z_ZIJIND], size_i)
    writeValue(edge_u_vals, entry[Z_UIJIND], uij, size_i)

    size_j = entry[Z_JLEN]
    uji = getValue(edge_u_vals, entry[Z_UJIIND], size_j) +\
          getValue(node_vals, entry[Z_XJIND], size_j) -\
          getValue(edge_z_vals, entry[Z_ZJIIND], size_j)
    writeValue(edge_u_vals, entry[Z_UJIIND], uji, size_j)
    return entry
