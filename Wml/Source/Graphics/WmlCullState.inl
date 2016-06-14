// Magic Software, Inc.
// http://www.magic-software.com
// http://www.wild-magic.com
// Copyright (c) 2004.  All Rights Reserved
//
// The Wild Magic Library (WML) source code is supplied under the terms of
// the license agreement http://www.magic-software.com/License/WildMagic.pdf
// and may not be copied or disclosed except in accordance with the terms of
// that agreement.

//----------------------------------------------------------------------------
inline bool& CullState::Enabled ()
{
    return m_bEnabled;
}
//----------------------------------------------------------------------------
inline CullState::FrontType& CullState::FrontFace ()
{
    return m_eFrontFace;
}
//----------------------------------------------------------------------------
inline CullState::CullType& CullState::CullFace ()
{
    return m_eCullFace;
}
//----------------------------------------------------------------------------


