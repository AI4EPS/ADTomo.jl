export eikonal

function eikonal(f::Union{Array{Float64}, PyObject},
    srcx::Int64,srcy::Int64,h::Float64)
    n_, m_ = size(f) # m width, n depth 
    n = n_-1
    m = m_-1
    eikonal_ = load_op_and_grad("$(@__DIR__)/build/libEikonal","eikonal")
    f,srcx,srcy,m,n,h = convert_to_tensor([f,srcx,srcy,m,n,h], [Float64,Int64,Int64,Int64,Int64,Float64])
    # f = tf.cast(f, dtype=tf.float64)
    # srcx = tf.cast(srcx, dtype=tf.int64)
    # srcy = tf.cast(srcy, dtype=tf.int64)
    # m = tf.cast(m, dtype=tf.int64)
    # n = tf.cast(n, dtype=tf.int64)
    # h = tf.cast(h, dtype=tf.float64)
    f = tf.reshape(f, (-1,))
    u = eikonal_(f,srcx,srcy,m,n,h)
    u.set_shape((length(f),))
    tf.reshape(u, (n_, m_))
end

