/* +---------------------------------------------------------------------------+
   |                     Mobile Robot Programming Toolkit (MRPT)               |
   |                          http://www.mrpt.org/                             |
   |                                                                           |
   | Copyright (c) 2005-2015, Individual contributors, see AUTHORS file        |
   | See: http://www.mrpt.org/Authors - All rights reserved.                   |
   | Released under BSD License. See details in http://www.mrpt.org/License    |
   +---------------------------------------------------------------------------+ */

#pragma once

#include <mrpt/utils/TColor.h>

namespace srba {

	/** @name Possible types for LANDMARK_TYPE::render_mode_t, used to control how a landmark is rendered 
	    @{ */

	/** Render 2D/3D point landmarks. */
	struct landmark_rendering_as_point 
	{   /** Parameters to be merged with RbaEngine<>::TOpenGLRepresentationOptions */
		struct TOpenGLRepresentationOptionsExtra
		{
		};
	};

	/** Render 2D/3D line landmarks. */
	struct landmark_rendering_as_line
	{   /** Parameters to be merged with RbaEngine<>::TOpenGLRepresentationOptions */
		struct TOpenGLRepresentationOptionsExtra
		{
		};
	};


	/** Render "fake graph-slam-like landmarks" as keyframe-to-keyframe constraints. */
	struct landmark_rendering_as_pose_constraints 
	{   /** Parameters to be merged with RbaEngine<>::TOpenGLRepresentationOptions */
		struct TOpenGLRepresentationOptionsExtra
		{
		};
	};

	/** Not to be rendered. */
	struct landmark_rendering_none 
	{   /** Parameters to be merged with RbaEngine<>::TOpenGLRepresentationOptions */
		struct TOpenGLRepresentationOptionsExtra
		{
		};
	};

	/** @} */

} // end NS

