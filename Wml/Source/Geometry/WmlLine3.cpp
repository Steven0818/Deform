// Magic Software, Inc.
// http://www.magic-software.com
// http://www.wild-magic.com
// Copyright (c) 2004.  All Rights Reserved
//
// The Wild Magic Library (WML) source code is supplied under the terms of
// the license agreement http://www.magic-software.com/License/WildMagic.pdf
// and may not be copied or disclosed except in accordance with the terms of
// that agreement.

#include "WmlLine3.h"
using namespace Wml;

//----------------------------------------------------------------------------
template <class Real>
Line3<Real>::Line3 ()
    :
    m_kOrigin(Vector3<Real>::ZERO),
    m_kDirection(Vector3<Real>::ZERO)
{
}
//----------------------------------------------------------------------------
template <class Real>
Vector3<Real>& Line3<Real>::Origin ()
{
    return m_kOrigin;
}
//----------------------------------------------------------------------------
template <class Real>
const Vector3<Real>& Line3<Real>::Origin () const
{
    return m_kOrigin;
}
//----------------------------------------------------------------------------
template <class Real>
Vector3<Real>& Line3<Real>::Direction ()
{
    return m_kDirection;
}
//----------------------------------------------------------------------------
template <class Real>
const Vector3<Real>& Line3<Real>::Direction () const
{
    return m_kDirection;
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// explicit instantiation
//----------------------------------------------------------------------------
namespace Wml
{
template class WML_ITEM Line3<float>;
template class WML_ITEM Line3<double>;
}
//----------------------------------------------------------------------------
