glcompat_attrs=21 //FLAG_TEXTURE+FLAG_LIGHT+FLAG_SPECULAR
class_mixedtexture=1

[VERTEX]

#include "common.glsl"

attribute mediump vec4 a_vert;
varying mediump vec4 v_light;
attribute vec3 a_norm;
varying mediump vec4 v_specular;
attribute vec4 a_texcoord;
varying mediump vec2 v_texcoord;

void main()
{
gl_Position = u_projmodmat * a_vert;
vec3 ecnormal=normalize(u_rotmat*a_norm);
v_light=max(dot(u_lightdir,ecnormal),0.0)*u_diffuse+u_ambient;
vec3 hv=normalize(-normalize(vec3(u_modmat*a_vert))+u_lightdir);
v_specular=u_speccolor*pow(max(dot(hv,ecnormal),0.0),u_shininess);
v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]

#include "common.glsl"

varying mediump vec4 v_light;
varying mediump vec4 v_specular;
varying mediump vec2 v_texcoord;

void main()
{
lowp vec4 c=u_color;
c*=mix(texture2D(u_tex,v_texcoord),texture2D(u_tex2,v_texcoord),u_mix);
c*=v_light;
c+=v_specular;
gl_FragColor=c;
}
