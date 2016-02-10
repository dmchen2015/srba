/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#pragma once


namespace srba {

/** Rebuild the Hessian symbolic information from the internal pointers to blocks of Jacobians.
	*  Only the upper triangle is filled-in (all what is needed for Cholesky) for square Hessians, in whole for rectangular ones (it depends on the symbolic decomposition, done elsewhere).
	* \tparam SPARSEBLOCKHESSIAN can be: TSparseBlocksHessian_6x6, TSparseBlocksHessian_3x3 or TSparseBlocksHessian_6x3
	* \return The number of Jacobian multiplications skipped due to its observation being marked as "invalid"
	*/
template <class KF2KF_POSE_TYPE,class LM_TYPE,class OBS_TYPE,class RBA_OPTIONS>
template <class SPARSEBLOCKHESSIAN>
size_t RbaEngine<KF2KF_POSE_TYPE,LM_TYPE,OBS_TYPE,RBA_OPTIONS>::sparse_hessian_update_numeric( SPARSEBLOCKHESSIAN & H ) const
{
	size_t nInvalid = 0;
	const size_t nUnknowns = H.getColCount();
	for (size_t i=0;i<nUnknowns;i++)
	{
		typename SPARSEBLOCKHESSIAN::col_t & col = H.getCol(i);

		for (typename SPARSEBLOCKHESSIAN::col_t::iterator it=col.begin();it!=col.end();++it)
		{
			typename SPARSEBLOCKHESSIAN::TEntry & entry = it->second;

			// Compute: Hij = \Sum_k  J_{ki}^t * \Lambda_k *  J_{kj}

			typename SPARSEBLOCKHESSIAN::matrix_t Hij;
			Hij.setZero();
			//const size_t nJacobs = entry.sym.lst_jacob_blocks.size();
			//for (size_t k=0;k<nJacobs;k++)
			const typename SPARSEBLOCKHESSIAN::symbolic_t::list_jacob_blocks_t::const_iterator itJ_end = entry.sym.lst_jacob_blocks.end();
			for (typename SPARSEBLOCKHESSIAN::symbolic_t::list_jacob_blocks_t::const_iterator itJ = entry.sym.lst_jacob_blocks.begin(); itJ!=itJ_end; ++itJ)
			{
				const typename SPARSEBLOCKHESSIAN::symbolic_t::THessianSymbolicInfoEntry & sym_k = *itJ;

				if (*sym_k.J1_valid && *sym_k.J2_valid)
				{
					// Accumulate Hessian sub-blocks:
					RBA_OPTIONS::obs_noise_matrix_t::template accum_JtJ(Hij, *sym_k.J1, *sym_k.J2, sym_k.obs_idx, this->parameters.obs_noise );
				}
				else nInvalid++;
			}

			// Do scaling (if applicable):
			RBA_OPTIONS::obs_noise_matrix_t::template scale_H(Hij, this->parameters.obs_noise );

			entry.num = Hij;
		}
	}
	return nInvalid;
} // end of sparse_hessian_update_numeric


// TODO: INCLUDE WEIGHTS IN THE HESSIAN UPDATING

template <class KF2KF_POSE_TYPE,class LM_TYPE,class OBS_TYPE,class RBA_OPTIONS>
template <class SPARSEBLOCKHESSIAN>
size_t RbaEngine<KF2KF_POSE_TYPE,LM_TYPE,OBS_TYPE,RBA_OPTIONS>::sparse_hessian_update_numeric( SPARSEBLOCKHESSIAN & H ,
																							   vector_weights_t & weights,
																							   double & stdv,
																							   const std::map<size_t,size_t> &obs_global_idx2residual_idx)
{
	size_t nInvalid = 0;

	this->parameters.obs_noise.std_noise_observations = stdv;

	const size_t OBS_DIMS  = rba_options_t::obs_noise_matrix_t::OBS_DIMS;
	Eigen::Matrix<double,OBS_DIMS,OBS_DIMS> W = Eigen::Matrix<double,OBS_DIMS,OBS_DIMS>::Identity();

	const size_t nUnknowns = H.getColCount();
	for (size_t i=0;i<nUnknowns;i++)
	{
		typename SPARSEBLOCKHESSIAN::col_t & col = H.getCol(i);

		for (typename SPARSEBLOCKHESSIAN::col_t::iterator it=col.begin();it!=col.end();++it)
		{
			typename SPARSEBLOCKHESSIAN::TEntry & entry = it->second;

			// Compute: Hij = \Sum_k  J_{ki}^t * \Lambda_k *  J_{kj}

			typename SPARSEBLOCKHESSIAN::matrix_t Hij;
			Hij.setZero();
			//const size_t nJacobs = entry.sym.lst_jacob_blocks.size();
			//for (size_t k=0;k<nJacobs;k++)
			const typename SPARSEBLOCKHESSIAN::symbolic_t::list_jacob_blocks_t::const_iterator itJ_end = entry.sym.lst_jacob_blocks.end();
			for (typename SPARSEBLOCKHESSIAN::symbolic_t::list_jacob_blocks_t::const_iterator itJ = entry.sym.lst_jacob_blocks.begin(); itJ!=itJ_end; ++itJ)
			{
				const typename SPARSEBLOCKHESSIAN::symbolic_t::THessianSymbolicInfoEntry & sym_k = *itJ;

				if (*sym_k.J1_valid && *sym_k.J2_valid)
				{
					//const size_t resid_idx = sequential_obs_indices[running_idx_obs++];
					std::map<size_t,size_t>::const_iterator it_obs = obs_global_idx2residual_idx.find(sym_k.obs_idx);
					ASSERT_(it_obs!=obs_global_idx2residual_idx.end())
					const size_t resid_idx = it_obs->second;

					// Accumulate Hessian sub-blocks:
					W(0,0) = weights[resid_idx](0);
					W(1,1) = weights[resid_idx](0);
					W(2,2) = weights[resid_idx](1);
					W(3,3) = weights[resid_idx](1);
					this->parameters.obs_noise.lambda = W;

					RBA_OPTIONS::obs_noise_matrix_t::template accum_JtJ(Hij, *sym_k.J1, *sym_k.J2, sym_k.obs_idx, this->parameters.obs_noise );
				}
				else nInvalid++;
			}

			// Do scaling (if applicable):
			RBA_OPTIONS::obs_noise_matrix_t::template scale_H(Hij, this->parameters.obs_noise );

			entry.num = Hij;
		}
	}

	// Reset default values
	this->parameters.obs_noise.std_noise_observations = 1.f;
	this->parameters.obs_noise.lambda = Eigen::Matrix<double,OBS_DIMS,OBS_DIMS>::Identity();

	return nInvalid;
} // end of sparse_hessian_update_numeric


} // end NS
