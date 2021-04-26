base_config = dict(
	factor_dim=16,
	residual_dim=128,

	residual_std=1,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size=16,
		n_epochs=200,

		learning_rate=dict(
			latent=1e-2,
			generator=1e-3,
			# encoder=1e-3,
			min=1e-5
		),

		loss_weights=dict(
			reconstruction=1,
			entropy=0.1,
			residual_decay=1e-3
		)
	)
)
