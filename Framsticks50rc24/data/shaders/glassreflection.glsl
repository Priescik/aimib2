glcompat_attrs=3
class_glcprogramext=1

[VERTEX]

#include "common.glsl"

attribute vec4 a_vert;
attribute vec3 a_norm;
varying mediump vec3 v_lnorm, v_refl;
void main()
{
	gl_Position = u_projmodmat * a_vert;
	v_lnorm = normalize(u_rotmat * a_norm);
	vec3 wnorm = normalize(mat3(u_viewmatinv) * v_lnorm);
	vec3 pos = (u_viewmatinv * (u_modmat * a_vert)).xyz;
	vec3 dir = u_campos - pos;
	v_refl = normalize(reflect(-dir, wnorm));
}

[FRAGMENT]

#include "common.glsl"

varying mediump vec3 v_lnorm, v_refl;

void main()
{
	mediump vec2 index;
	index.x = atan(-v_refl.x, -v_refl.y) / 6.283;
	index.y = clamp((-asin(v_refl.z) / 3.15) + 0.5, 0.1, 0.9);
	lowp vec4 t = texture2D(u_tex, index);
	t.a = max(pow(1.0 - v_lnorm.z, 0.5), t.a);
	gl_FragColor = t;
}
