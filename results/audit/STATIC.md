# Static audit: provided graphs vs callproof graphs

127 graphs (one per distinct base commit). Percentages are micro-averages over all graphs.
Mean build time 3.8s per checkout (max 8.1s, under parallel load).
Provided embeddings are compressed: the median cosine similarity between random node pairs is 0.881 (mean over graphs).
Edges are checked against the source text of their endpoints: an edge u -> v is *supported* when u's
code contains v (class -> method), inherits from, instantiates, calls or references v by name. The check
is name-based, so resolved edges it cannot see (inherited `__init__`, `cls(...)`, `obj(...)` -> `__call__`)
count as unsupported.

| metric | provided | callproof |
|---|---|---|
| nodes | 367985 | 534246 |
| edges | 452588 | 1398555 |
| definitions with a node | 84.3% (376726/446843) | 100.0% (446843/446843) |
| async definitions with a node | 0.0% (0/70117) | 100.0% (70117/70117) |
| modules with a node | 0.0% (0/88857) | 100.0% (88857/88857) |
| definitions sharing an id | 3.0% (13184/446843) | 0.6% (2535/446843) |
| definitions whose id is not module + dotted qualname | 5.6% (24882/446843) | 0.0% (0/446843) |
| top-level-class methods whose id lost the class | 2.4% (1425/60632) | 0.0% (0/60632) |
| nodes with >50 neighbours (truncated by get_code_neighbors) | 0.1% (327/367985) | 0.5% (2581/534246) |
| connected pairs also linked in reverse | 100.0% (449902/449902) | 0.8% (9926/1254148) |
| callable nodes in no call relation | 63.7% (191433/300316) | 57.8% (217087/375292) |
| ... library code only | 48.8% (30129/61730) | 9.0% (5860/65105) |
| checked edges whose source supports their direction | 50.9% (229132/449902) | 95.3% (562617/590619) |
| checked edges supported only in reverse | 48.7% (219226/449902) | 0.3% (1762/590619) |
| checked edges unsupported by source text | 0.3% (1544/449902) | 4.4% (26240/590619) |
| nodes with an embedding | 100.0% (367985/367985) | 65.6% (350393/534246) |

Provided-graph pair semantics (unordered connected pairs, strongest explanation):

- contains: 26.4% (59477/224951)
- inherits: 5.6% (12673/224951)
- instantiates: 39.6% (89086/224951)
- calls: 27.4% (61712/224951)
- references: 0.5% (1231/224951)
- unexplained: 0.3% (772/224951)

Total provided pairs: 224951. Callproof edge types: calls=592421, contains=445389, decorated_by=95724, imports=165580, inherits=13126, references=86315. Edge resolutions: override=3277, protocol=20116, static=1372597, unique-name=2565.

Random provided edges for hand inspection (30 of 254 sampled):

| source | target | relation u -> v | relation v -> u |
|---|---|---|---|
| `custom_request_and_route.tutorial003_py310.TimedRoute` | `fastapi.routing.APIRoute` | inherits | - |
| `fastapi.datastructures.DefaultPlaceholder` | `fastapi.datastructures.DefaultPlaceholder.__bool__` | contains | - |
| `fastapi.datastructures.UploadFile` | `fastapi.datastructures.UploadFile.__get_pydantic_json_schema__` | contains | - |
| `fastapi.exceptions.ValidationException.errors` | `fastapi.exceptions.ValidationException` | - | contains |
| `fastapi.param_functions.Form` | `fastapi.security.oauth2.OAuth2PasswordRequestForm.__init__` | - | calls |
| `fastapi.routing.APIWebSocketRoute` | `fastapi.routing.APIWebSocketRoute.matches` | contains | - |
| `fastapi.security.oauth2.OAuth2PasswordBearer.__init__` | `fastapi.security.oauth2.OAuth2PasswordBearer` | - | contains |
| `requests.cookies.RequestsCookieJar.multiple_domains` | `requests.cookies.RequestsCookieJar` | - | contains |
| `requests.sessions.Session.__init__` | `requests.sessions.Session` | - | contains |
| `requests.sessions.SessionRedirectMixin` | `requests.sessions.Session` | - | inherits |
| `rich._win32_console.LegacyWindowsTerm` | `rich._win32_console.LegacyWindowsTerm.erase_line` | contains | - |
| `rich.console.Console` | `tests.test_prompt.test_prompt_confirm_default` | - | instantiates |
| `rich.markdown.ListItem.__init__` | `rich.markdown.ListItem` | - | contains |
| `rich.padding.Padding.unpack` | `rich.table.Table.padding` | - | calls |
| `rich.progress.SpinnerColumn` | `rich.progress.SpinnerColumn.__init__` | contains | - |
| `rich.segment.Segment` | `tests.test_segment.test_filter_control` | - | instantiates |
| `scripts.docs.VisibleTextExtractor` | `scripts.docs.VisibleTextExtractor.extract_visible_text` | contains | - |
| `tests.test_color.test_parse_rgb_hex` | `rich.color_triplet.ColorTriplet` | instantiates | - |
| `tests.test_compat.test_v1_plain_validator_function` | `fastapi._compat.v1.with_info_plain_validator_function` | calls | - |
| `tests.test_inherited_custom_class.MyUuid` | `tests.test_inherited_custom_class.return_some_user` | - | instantiates |
| `tests.test_inspect.test_object_types_mro_as_strings` | `rich._inspect.get_object_types_mro_as_strings` | calls | - |
| `tests.test_markup.test_render_escape` | `rich.markup.escape` | calls | - |
| `tests.test_openapi_separate_input_output_schemas.test_read_items` | `tests.test_openapi_separate_input_output_schemas.get_app_client` | calls | - |
| `tests.test_pydantic_v1_v2_noneable.NewSubItem` | `tests.test_pydantic_v1_v2_noneable.handle_v1_item_to_v2` | - | instantiates |
| `tests.test_requests.TestCaseInsensitiveDict.test_len` | `tests.test_requests.TestCaseInsensitiveDict` | - | contains |
| `tests.test_requests.TestRequests.test_respect_proxy_env_on_send_with_redirects` | `tests.test_requests.TestRequests` | - | contains |
| `tests.test_response_model_include_exclude.Model2` | `tests.test_response_model_include_exclude.mixed` | - | instantiates |
| `tests.test_syntax.test_python_render_simple` | `tests.render.render` | calls | - |
| `tests.test_theme.test_theme_stack` | `rich.style.Style.parse` | calls | - |
| `tests.test_utils.LenFile` | `tests.test_utils.LenFile.__init__` | contains | - |

## Agreement with traced executions (129 tasks with ground truth)

Dynamic edges are compared at the tracer's granularity (nested functions folded into their
enclosing named definition). Categories overlap.

| metric | provided | callproof (all edges) | callproof (calls only) |
|---|---|---|---|
| gold modified definitions with a node | 91.1% (452/496) | 100.0% (496/496) |  |
| gold added definitions with a node | 1.9% (5/267) | 1.9% (5/267) |  |
| gold modified module-level edits with a node | 0.0% (0/112) | 100.0% (112/112) |  |
| executed definitions with a node | 87.6% (6105/6966) | 91.8% (6395/6966) |  |
| executed (incl. module code) with a node | 86.5% (6105/7055) | 91.5% (6457/7055) |  |
| F2P tests present at base with a node | 54.3% (496/914) | 54.3% (496/914) |  |
| dynamic edges [all]: both ends are nodes | 86.2% (8254/9576) | 89.9% (8607/9576) |  |
| dynamic edges [all]: linked, either direction | 25.9% (2480/9576) | 79.9% (7649/9576) | 79.6% (7626/9576) |
| dynamic edges [all]: linked, correct direction | 25.9% (2480/9576) | 79.3% (7590/9576) | 79.0% (7569/9576) |
| dynamic edges [self_method]: both ends are nodes | 97.8% (2405/2459) | 98.3% (2416/2459) |  |
| dynamic edges [self_method]: linked, either direction | 0.0% (0/2459) | 93.0% (2288/2459) | 93.0% (2288/2459) |
| dynamic edges [self_method]: linked, correct direction | 0.0% (0/2459) | 92.8% (2282/2459) | 92.8% (2282/2459) |
| dynamic edges [constructor]: both ends are nodes | 79.9% (873/1093) | 80.6% (881/1093) |  |
| dynamic edges [constructor]: linked, either direction | 0.0% (0/1093) | 74.3% (812/1093) | 74.1% (810/1093) |
| dynamic edges [constructor]: linked, correct direction | 0.0% (0/1093) | 74.3% (812/1093) | 74.1% (810/1093) |
| dynamic edges [module_function]: both ends are nodes | 88.5% (2685/3034) | 97.6% (2962/3034) |  |
| dynamic edges [module_function]: linked, either direction | 74.3% (2255/3034) | 93.7% (2844/3034) | 93.7% (2842/3034) |
| dynamic edges [module_function]: linked, correct direction | 74.3% (2255/3034) | 93.6% (2840/3034) | 93.5% (2838/3034) |
| dynamic edges [cross_module]: both ends are nodes | 82.1% (4080/4970) | 85.2% (4236/4970) |  |
| dynamic edges [cross_module]: linked, either direction | 31.9% (1586/4970) | 73.0% (3627/4970) | 72.6% (3609/4970) |
| dynamic edges [cross_module]: linked, correct direction | 31.9% (1586/4970) | 71.9% (3575/4970) | 71.6% (3558/4970) |
| dynamic edges [async]: both ends are nodes | 0.0% (0/409) | 84.6% (346/409) |  |
| dynamic edges [async]: linked, either direction | 0.0% (0/409) | 72.4% (296/409) | 72.4% (296/409) |
| dynamic edges [async]: linked, correct direction | 0.0% (0/409) | 72.4% (296/409) | 72.4% (296/409) |
| dynamic edges [via_test]: both ends are nodes | 50.0% (760/1520) | 50.9% (773/1520) |  |
| dynamic edges [via_test]: linked, either direction | 6.0% (91/1520) | 43.8% (665/1520) | 43.7% (664/1520) |
| dynamic edges [via_test]: linked, correct direction | 6.0% (91/1520) | 43.5% (661/1520) | 43.4% (660/1520) |
| dynamic edges [nested]: both ends are nodes | 70.2% (575/819) | 85.2% (698/819) |  |
| dynamic edges [nested]: linked, either direction | 17.5% (143/819) | 64.3% (527/819) | 64.0% (524/819) |
| dynamic edges [nested]: linked, correct direction | 17.5% (143/819) | 60.3% (494/819) | 60.2% (493/819) |
| constructor calls linked to the __init__ or to the class | 67.0% (732/1093) | 78.9% (862/1093) | 74.2% (811/1093) |
| tasks with an F2P test present at base | 31.4% (33/105) | 31.4% (33/105) |  |
| gold modified defs within 1 hops of an F2P test (tasks with one) | 3.0% (9/303) | 4.3% (13/303) | 4.3% (13/303) |
| gold modified defs within 2 hops of an F2P test (tasks with one) | 8.6% (26/303) | 22.1% (67/303) | 8.3% (25/303) |
| gold modified defs within 3 hops of an F2P test (tasks with one) | 15.2% (46/303) | 47.9% (145/303) | 25.4% (77/303) |
| tasks with executed test code present at base | 36.2% (38/105) | 66.7% (70/105) |  |
| gold modified defs within 1 hops of executed test code | 2.9% (9/308) | 3.2% (13/404) | n/a |
| gold modified defs within 2 hops of executed test code | 8.4% (26/308) | 16.6% (67/404) | n/a |
| gold modified defs within 3 hops of executed test code | 15.9% (49/308) | 36.6% (148/404) | n/a |
