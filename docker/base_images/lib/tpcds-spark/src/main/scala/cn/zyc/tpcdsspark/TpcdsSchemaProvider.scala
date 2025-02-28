package cn.zyc.tpcdsspark

import org.apache.spark.SparkContext

case class Catalog_sales(
                          cs_sold_date_sk: Long,
                          cs_sold_time_sk: Long,
                          cs_ship_date_sk: Long,
                          cs_bill_customer_sk: Long,
                          cs_bill_cdemo_sk: Long,
                          cs_bill_hdemo_sk: Long,
                          cs_bill_addr_sk: Long,
                          cs_ship_customer_sk: Long,
                          cs_ship_cdemo_sk: Long,
                          cs_ship_hdemo_sk: Long,
                          cs_ship_addr_sk: Long,
                          cs_call_center_sk: Long,
                          cs_catalog_page_sk: Long,
                          cs_ship_mode_sk: Long,
                          cs_warehouse_sk: Long,
                          cs_item_sk: Long,
                          cs_promo_sk: Long,
                          cs_order_number: Long,
                          cs_quantity: Long,
                          cs_wholesale_cost: Double,
                          cs_list_price: Double,
                          cs_sales_price: Double,
                          cs_ext_discount_amt: Double,
                          cs_ext_sales_price: Double,
                          cs_ext_wholesale_cost: Double,
                          cs_ext_list_price: Double,
                          cs_ext_tax: Double,
                          cs_coupon_amt: Double,
                          cs_ext_ship_cost: Double,
                          cs_net_paid: Double,
                          cs_net_paid_inc_tax: Double,
                          cs_net_paid_inc_ship: Double,
                          cs_net_paid_inc_ship_tax: Double,
                          cs_net_profit: Double)

case class Catalog_returns(
                            cr_returned_date_sk: Long,
                            cr_returned_time_sk: Long,
                            cr_item_sk: Long,
                            cr_refunded_customer_sk: Long,
                            cr_refunded_cdemo_sk: Long,
                            cr_refunded_hdemo_sk: Long,
                            cr_refunded_addr_sk: Long,
                            cr_returning_customer_sk: Long,
                            cr_returning_cdemo_sk: Long,
                            cr_returning_hdemo_sk: Long,
                            cr_returning_addr_sk: Long,
                            cr_call_center_sk: Long,
                            cr_catalog_page_sk: Long,
                            cr_ship_mode_sk: Long,
                            cr_warehouse_sk: Long,
                            cr_reason_sk: Long,
                            cr_order_number: Long,
                            cr_return_quantity: Long,
                            cr_return_amount: Double,
                            cr_return_tax: Double,
                            cr_return_amt_inc_tax: Double,
                            cr_fee: Double,
                            cr_return_ship_cost: Double,
                            cr_refunded_cash: Double,
                            cr_reversed_charge: Double,
                            cr_store_credit: Double,
                            cr_net_loss: Double)

case class Inventory(
                      inv_date_sk: Long,
                      inv_item_sk: Long,
                      inv_warehouse_sk: Long,
                      inv_quantity_on_hand: Long)

case class Store_sales(
                        ss_sold_date_sk: Long,
                        ss_sold_time_sk: Long,
                        ss_item_sk: Long,
                        ss_customer_sk: Long,
                        ss_cdemo_sk: Long,
                        ss_hdemo_sk: Long,
                        ss_addr_sk: Long,
                        ss_store_sk: Long,
                        ss_promo_sk: Long,
                        ss_ticket_number: Long,
                        ss_quantity: Long,
                        ss_wholesale_cost: Double,
                        ss_list_price: Double,
                        ss_sales_price: Double,
                        ss_ext_discount_amt: Double,
                        ss_ext_sales_price: Double,
                        ss_ext_wholesale_cost: Double,
                        ss_ext_list_price: Double,
                        ss_ext_tax: Double,
                        ss_coupon_amt: Double,
                        ss_net_paid: Double,
                        ss_net_paid_inc_tax: Double,
                        ss_net_profit: Double)

case class Store_returns(
                          sr_returned_date_sk: Long,
                          sr_return_time_sk: Long,
                          sr_item_sk: Long,
                          sr_customer_sk: Long,
                          sr_cdemo_sk: Long,
                          sr_hdemo_sk: Long,
                          sr_addr_sk: Long,
                          sr_store_sk: Long,
                          sr_reason_sk: Long,
                          sr_ticket_number: Long,
                          sr_return_quantity: Long,
                          sr_return_amt: Double,
                          sr_return_tax: Double,
                          sr_return_amt_inc_tax: Double,
                          sr_fee: Double,
                          sr_return_ship_cost: Double,
                          sr_refunded_cash: Double,
                          sr_reversed_charge: Double,
                          sr_store_credit: Double,
                          sr_net_loss: Double)

case class Web_sales(
                      ws_sold_date_sk: Long,
                      ws_sold_time_sk: Long,
                      ws_ship_date_sk: Long,
                      ws_item_sk: Long,
                      ws_bill_customer_sk: Long,
                      ws_bill_cdemo_sk: Long,
                      ws_bill_hdemo_sk: Long,
                      ws_bill_addr_sk: Long,
                      ws_ship_customer_sk: Long,
                      ws_ship_cdemo_sk: Long,
                      ws_ship_hdemo_sk: Long,
                      ws_ship_addr_sk: Long,
                      ws_web_page_sk: Long,
                      ws_web_site_sk: Long,
                      ws_ship_mode_sk: Long,
                      ws_warehouse_sk: Long,
                      ws_promo_sk: Long,
                      ws_order_number: Long,
                      ws_quantity: Long,
                      ws_wholesale_cost: Double,
                      ws_list_price: Double,
                      ws_sales_price: Double,
                      ws_ext_discount_amt: Double,
                      ws_ext_sales_price: Double,
                      ws_ext_wholesale_cost: Double,
                      ws_ext_list_price: Double,
                      ws_ext_tax: Double,
                      ws_coupon_amt: Double,
                      ws_ext_ship_cost: Double,
                      ws_net_paid: Double,
                      ws_net_paid_inc_tax: Double,
                      ws_net_paid_inc_ship: Double,
                      ws_net_paid_inc_ship_tax: Double,
                      ws_net_profit: Double)

case class Web_returns(
                        wr_returned_date_sk: Long,
                        wr_returned_time_sk: Long,
                        wr_item_sk: Long,
                        wr_refunded_customer_sk: Long,
                        wr_refunded_cdemo_sk: Long,
                        wr_refunded_hdemo_sk: Long,
                        wr_refunded_addr_sk: Long,
                        wr_returning_customer_sk: Long,
                        wr_returning_cdemo_sk: Long,
                        wr_returning_hdemo_sk: Long,
                        wr_returning_addr_sk: Long,
                        wr_web_page_sk: Long,
                        wr_reason_sk: Long,
                        wr_order_number: Long,
                        wr_return_quantity: Long,
                        wr_return_amt: Double,
                        wr_return_tax: Double,
                        wr_return_amt_inc_tax: Double,
                        wr_fee: Double,
                        wr_return_ship_cost: Double,
                        wr_refunded_cash: Double,
                        wr_reversed_charge: Double,
                        wr_account_credit: Double,
                        wr_net_loss: Double)

case class Call_center(
                        cc_call_center_sk: Long,
                        cc_call_center_id: String,
                        cc_rec_start_date: String,
                        cc_rec_end_date: String,
                        cc_closed_date_sk: Long,
                        cc_open_date_sk: Long,
                        cc_name: String,
                        cc_class: String,
                        cc_employees: Long,
                        cc_sq_ft: Long,
                        cc_hours: String,
                        cc_manager: String,
                        cc_mkt_id: Long,
                        cc_mkt_class: String,
                        cc_mkt_desc: String,
                        cc_market_manager: String,
                        cc_division: Long,
                        cc_division_name: String,
                        cc_company: Long,
                        cc_company_name: String,
                        cc_street_number: String,
                        cc_street_name: String,
                        cc_street_type: String,
                        cc_suite_number: String,
                        cc_city: String,
                        cc_county: String,
                        cc_state: String,
                        cc_zip: String,
                        cc_country: String,
                        cc_gmt_offset: Double,
                        cc_tax_percentage: Double)

case class Catalog_page(
                         cp_catalog_page_sk: Long,
                         cp_catalog_page_id: String,
                         cp_start_date_sk: Long,
                         cp_end_date_sk: Long,
                         cp_department: String,
                         cp_catalog_number: Long,
                         cp_catalog_page_number: Long,
                         cp_description: String,
                         cp_type: String)

case class Customer(
                     c_customer_sk: Long,
                     c_customer_id: String,
                     c_current_cdemo_sk: Long,
                     c_current_hdemo_sk: Long,
                     c_current_addr_sk: Long,
                     c_first_shipto_date_sk: Long,
                     c_first_sales_date_sk: Long,
                     c_salutation: String,
                     c_first_name: String,
                     c_last_name: String,
                     c_preferred_cust_flag: String,
                     c_birth_day: Long,
                     c_birth_month: Long,
                     c_birth_year: Long,
                     c_birth_country: String,
                     c_login: String,
                     c_email_address: String,
                     c_last_review_date: String)

case class Customer_address(
                             ca_address_sk: Long,
                             ca_address_id: String,
                             ca_street_number: String,
                             ca_street_name: String,
                             ca_street_type: String,
                             ca_suite_number: String,
                             ca_city: String,
                             ca_county: String,
                             ca_state: String,
                             ca_zip: String,
                             ca_country: String,
                             ca_gmt_offset: Double,
                             ca_location_type: String)

case class Customer_demographics(
                                  cd_demo_sk: Long,
                                  cd_gender: String,
                                  cd_marital_status: String,
                                  cd_education_status: String,
                                  cd_purchase_estimate: Long,
                                  cd_credit_rating: String,
                                  cd_dep_count: Long,
                                  cd_dep_employed_count: Long,
                                  cd_dep_college_count: Long)

case class Date_dim(
                     d_date_sk: Long,
                     d_date_id: String,
                     d_date: String,
                     d_month_seq: Long,
                     d_week_seq: Long,
                     d_quarter_seq: Long,
                     d_year: Long,
                     d_dow: Long,
                     d_moy: Long,
                     d_dom: Long,
                     d_qoy: Long,
                     d_fy_year: Long,
                     d_fy_quarter_seq: Long,
                     d_fy_week_seq: Long,
                     d_day_name: String,
                     d_quarter_name: String,
                     d_holiday: String,
                     d_weekend: String,
                     d_following_holiday: String,
                     d_first_dom: Long,
                     d_last_dom: Long,
                     d_same_day_ly: Long,
                     d_same_day_lq: Long,
                     d_current_day: String,
                     d_current_week: String,
                     d_current_month: String,
                     d_current_quarter: String,
                     d_current_year: String)

case class Household_demographics(
                                   hd_demo_sk: Long,
                                   hd_income_band_sk: Long,
                                   hd_buy_potential: String,
                                   hd_dep_count: Long,
                                   hd_vehicle_count: Long)

case class Income_band(
                        ib_income_band_sk: Long,
                        ib_lower_bound: Long,
                        ib_upper_bound: Long)

case class Item(
                 i_item_sk: Long,
                 i_item_id: String,
                 i_rec_start_date: String,
                 i_rec_end_date: String,
                 i_item_desc: String,
                 i_current_price: Double,
                 i_wholesale_cost: Double,
                 i_brand_id: Long,
                 i_brand: String,
                 i_class_id: Long,
                 i_class: String,
                 i_category_id: Long,
                 i_category: String,
                 i_manufact_id: Long,
                 i_manufact: String,
                 i_size: String,
                 i_formulation: String,
                 i_color: String,
                 i_units: String,
                 i_container: String,
                 i_manager_id: Long,
                 i_product_name: String)

case class Promotion(
                      p_promo_sk: Long,
                      p_promo_id: String,
                      p_start_date_sk: Long,
                      p_end_date_sk: Long,
                      p_item_sk: Long,
                      p_cost: Double,
                      p_response_target: Long,
                      p_promo_name: String,
                      p_channel_dmail: String,
                      p_channel_email: String,
                      p_channel_catalog: String,
                      p_channel_tv: String,
                      p_channel_radio: String,
                      p_channel_press: String,
                      p_channel_event: String,
                      p_channel_demo: String,
                      p_channel_details: String,
                      p_purpose: String,
                      p_discount_active: String)

case class Reason(
                   r_reason_sk: Long,
                   r_reason_id: String,
                   r_reason_desc: String)

case class Ship_mode(
                      sm_ship_mode_sk: Long,
                      sm_ship_mode_id: String,
                      sm_type: String,
                      sm_code: String,
                      sm_carrier: String,
                      sm_contract: String)

case class Store(
                  s_store_sk: Long,
                  s_store_id: String,
                  s_rec_start_date: String,
                  s_rec_end_date: String,
                  s_closed_date_sk: Long,
                  s_store_name: String,
                  s_number_employees: Long,
                  s_floor_space: Long,
                  s_hours: String,
                  s_manager: String,
                  s_market_id: Long,
                  s_geography_class: String,
                  s_market_desc: String,
                  s_market_manager: String,
                  s_division_id: Long,
                  s_division_name: String,
                  s_company_id: Long,
                  s_company_name: String,
                  s_street_number: String,
                  s_street_name: String,
                  s_street_type: String,
                  s_suite_number: String,
                  s_city: String,
                  s_county: String,
                  s_state: String,
                  s_zip: String,
                  s_country: String,
                  s_gmt_offset: Double,
                  s_tax_precentage: Double)

case class Time_dim(
                     t_time_sk: Long,
                     t_time_id: String,
                     t_time: Long,
                     t_hour: Long,
                     t_minute: Long,
                     t_second: Long,
                     t_am_pm: String,
                     t_shift: String,
                     t_sub_shift: String,
                     t_meal_time: String)

case class Warehouse(
                      w_warehouse_sk: Long,
                      w_warehouse_id: String,
                      w_warehouse_name: String,
                      w_warehouse_sq_ft: Long,
                      w_street_number: String,
                      w_street_name: String,
                      w_street_type: String,
                      w_suite_number: String,
                      w_city: String,
                      w_county: String,
                      w_state: String,
                      w_zip: String,
                      w_country: String,
                      w_gmt_offset: Double)

case class Web_page(
                     wp_web_page_sk: Long,
                     wp_web_page_id: String,
                     wp_rec_start_date: String,
                     wp_rec_end_date: String,
                     wp_creation_date_sk: Long,
                     wp_access_date_sk: Long,
                     wp_autogen_flag: String,
                     wp_customer_sk: Long,
                     wp_url: String,
                     wp_type: String,
                     wp_char_count: Long,
                     wp_link_count: Long,
                     wp_image_count: Long,
                     wp_max_ad_count: Long)

case class Web_site(
                     web_site_sk: Long,
                     web_site_id: String,
                     web_rec_start_date: String,
                     web_rec_end_date: String,
                     web_name: String,
                     web_open_date_sk: Long,
                     web_close_date_sk: Long,
                     web_class: String,
                     web_manager: String,
                     web_mkt_id: Long,
                     web_mkt_class: String,
                     web_mkt_desc: String,
                     web_market_manager: String,
                     web_company_id: Long,
                     web_company_name: String,
                     web_street_number: String,
                     web_street_name: String,
                     web_street_type: String,
                     web_suite_number: String,
                     web_city: String,
                     web_county: String,
                     web_state: String,
                     web_zip: String,
                     web_country: String,
                     web_gmt_offset: String,
                     web_tax_percentage: Double)

class TpcdsSchemaProvider(sc: SparkContext, inputDir: String) extends java.io.Serializable{
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._


  val len_map = Map(
    "catalog_sales" -> 34,
    "catalog_returns" -> 27,
    "inventory" -> 4,
    "store_sales" -> 23,
    "store_returns" -> 20,
    "web_sales" -> 34,
    "web_returns" -> 24,
    "call_center" -> 31,
    "catalog_page" -> 9,
    "customer" -> 18,
    "customer_address" -> 13,
    "customer_demographics" -> 9,
    "date_dim" -> 28,
    "household_demographics" -> 5,
    "income_band" -> 3,
    "item" -> 22,
    "promotion" -> 19,
    "reason" -> 3,
    "ship_mode" -> 6,
    "store" -> 29,
    "time_dim" -> 10,
    "warehouse" -> 14,
    "web_page" -> 14,
    "web_site" -> 26 )
  def judge(p:Array[String], table_name: String) :Boolean = {
    if(p.contains("") || p.size != len_map(table_name))
      false
    else
      true
  }

  val dfMap = Map(
    "catalog_sales" -> sc.textFile(inputDir + "/catalog_sales.dat").map(_.split('|')).filter(p => judge(p, "catalog_sales")).map(p =>
      Catalog_sales(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong, p(14).trim.toLong, p(15).trim.toLong, p(16).trim.toLong, p(17).trim.toLong, p(18).trim.toLong, p(19).trim.toDouble, p(20).trim.toDouble, p(21).trim.toDouble, p(22).trim.toDouble, p(23).trim.toDouble, p(24).trim.toDouble, p(25).trim.toDouble, p(26).trim.toDouble, p(27).trim.toDouble, p(28).trim.toDouble, p(29).trim.toDouble, p(30).trim.toDouble, p(31).trim.toDouble, p(32).trim.toDouble, p(33).trim.toDouble)).toDF(),

    "catalog_returns" -> sc.textFile(inputDir + "/catalog_returns.dat").map(_.split('|')).filter(p => judge(p, "catalog_returns")).map(p =>
      Catalog_returns(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong, p(14).trim.toLong, p(15).trim.toLong, p(16).trim.toLong, p(17).trim.toLong, p(18).trim.toDouble, p(19).trim.toDouble, p(20).trim.toDouble, p(21).trim.toDouble, p(22).trim.toDouble, p(23).trim.toDouble, p(24).trim.toDouble, p(25).trim.toDouble, p(26).trim.toDouble)).toDF(),

    "inventory" -> sc.textFile(inputDir + "/inventory.dat").map(_.split('|')).filter(p => judge(p, "inventory")).map(p =>
      Inventory(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong)).toDF(),

    "store_sales" -> sc.textFile(inputDir + "/store_sales.dat").map(_.split('|')).filter(p => judge(p, "store_sales")).map(p =>
      Store_sales(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toDouble, p(12).trim.toDouble, p(13).trim.toDouble, p(14).trim.toDouble, p(15).trim.toDouble, p(16).trim.toDouble, p(17).trim.toDouble, p(18).trim.toDouble, p(19).trim.toDouble, p(20).trim.toDouble, p(21).trim.toDouble, p(22).trim.toDouble)).toDF(),

    "store_returns" -> sc.textFile(inputDir + "/store_returns.dat").map(_.split('|')).filter(p => judge(p, "store_returns")).map(p =>
      Store_returns(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toDouble, p(12).trim.toDouble, p(13).trim.toDouble, p(14).trim.toDouble, p(15).trim.toDouble, p(16).trim.toDouble, p(17).trim.toDouble, p(18).trim.toDouble, p(19).trim.toDouble)).toDF(),

    "web_sales" -> sc.textFile(inputDir + "/web_sales.dat").map(_.split('|')).filter(p => judge(p, "web_sales")).map(p =>
      Web_sales(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong, p(14).trim.toLong, p(15).trim.toLong, p(16).trim.toLong, p(17).trim.toLong, p(18).trim.toLong, p(19).trim.toDouble, p(20).trim.toDouble, p(21).trim.toDouble, p(22).trim.toDouble, p(23).trim.toDouble, p(24).trim.toDouble, p(25).trim.toDouble, p(26).trim.toDouble, p(27).trim.toDouble, p(28).trim.toDouble, p(29).trim.toDouble, p(30).trim.toDouble, p(31).trim.toDouble, p(32).trim.toDouble, p(33).trim.toDouble)).toDF(),

    "web_returns" -> sc.textFile(inputDir + "/web_returns.dat").map(_.split('|')).filter(p => judge(p, "web_returns")).map(p =>
      Web_returns(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong, p(14).trim.toLong, p(15).trim.toDouble, p(16).trim.toDouble, p(17).trim.toDouble, p(18).trim.toDouble, p(19).trim.toDouble, p(20).trim.toDouble, p(21).trim.toDouble, p(22).trim.toDouble, p(23).trim.toDouble)).toDF(),

    "call_center" -> sc.textFile(inputDir + "/call_center.dat").map(_.split('|')).filter(p => judge(p, "call_center")).map(p =>
      Call_center(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim.toLong, p(5).trim.toLong, p(6).trim, p(7).trim, p(8).trim.toLong, p(9).trim.toLong, p(10).trim, p(11).trim, p(12).trim.toLong, p(13).trim, p(14).trim, p(15).trim, p(16).trim.toLong, p(17).trim, p(18).trim.toLong, p(19).trim, p(20).trim, p(21).trim, p(22).trim, p(23).trim, p(24).trim, p(25).trim, p(26).trim, p(27).trim, p(28).trim, p(29).trim.toDouble, p(30).trim.toDouble)).toDF(),

    "catalog_page" -> sc.textFile(inputDir + "/catalog_page.dat").map(_.split('|')).filter(p => judge(p, "catalog_page")).map(p =>
      Catalog_page(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim.toLong, p(4).trim, p(5).trim.toLong, p(6).trim.toLong, p(7).trim, p(8).trim)).toDF(),

    "customer" -> sc.textFile(inputDir + "/customer.dat").map(_.split('|')).filter(p => judge(p, "customer")).map(p =>
      Customer(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim, p(8).trim, p(9).trim, p(10).trim, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong, p(14).trim, p(15).trim, p(16).trim, p(17).trim)).toDF(),

    "customer_address" -> sc.textFile(inputDir + "/customer_address.dat").map(_.split('|')).filter(p => judge(p, "customer_address")).map(p =>
      Customer_address(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim, p(5).trim, p(6).trim, p(7).trim, p(8).trim, p(9).trim, p(10).trim, p(11).trim.toDouble, p(12).trim)).toDF(),

    "customer_demographics" -> sc.textFile(inputDir + "/customer_demographics.dat").map(_.split('|')).filter(p => judge(p, "customer_demographics")).map(p =>
      Customer_demographics(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim.toLong, p(5).trim, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong)).toDF(),

    "date_dim" -> sc.textFile(inputDir + "/date_dim.dat").map(_.split('|')).filter(p => judge(p, "date_dim")).map(p =>
      Date_dim(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim.toLong, p(7).trim.toLong, p(8).trim.toLong, p(9).trim.toLong, p(10).trim.toLong, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong, p(14).trim, p(15).trim, p(16).trim, p(17).trim, p(18).trim, p(19).trim.toLong, p(20).trim.toLong, p(21).trim.toLong, p(22).trim.toLong, p(23).trim, p(24).trim, p(25).trim, p(26).trim, p(27).trim)).toDF(),

    "household_demographics" -> sc.textFile(inputDir + "/household_demographics.dat").map(_.split('|')).filter(p => judge(p, "household_demographics")).map(p =>
      Household_demographics(p(0).trim.toLong, p(1).trim.toLong, p(2).trim, p(3).trim.toLong, p(4).trim.toLong)).toDF(),

    "income_band" -> sc.textFile(inputDir + "/income_band.dat").map(_.split('|')).filter(p => judge(p, "income_band")).map(p =>
      Income_band(p(0).trim.toLong, p(1).trim.toLong, p(2).trim.toLong)).toDF(),

    "item" -> sc.textFile(inputDir + "/item.dat").map(_.split('|')).filter(p => judge(p, "item")).map(p =>
      Item(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim, p(5).trim.toDouble, p(6).trim.toDouble, p(7).trim.toLong, p(8).trim, p(9).trim.toLong, p(10).trim, p(11).trim.toLong, p(12).trim, p(13).trim.toLong, p(14).trim, p(15).trim, p(16).trim, p(17).trim, p(18).trim, p(19).trim, p(20).trim.toLong, p(21).trim)).toDF(),

    "promotion" -> sc.textFile(inputDir + "/promotion.dat").map(_.split('|')).filter(p => judge(p, "promotion")).map(p =>
      Promotion(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toDouble, p(6).trim.toLong, p(7).trim, p(8).trim, p(9).trim, p(10).trim, p(11).trim, p(12).trim, p(13).trim, p(14).trim, p(15).trim, p(16).trim, p(17).trim, p(18).trim)).toDF(),

    "reason" -> sc.textFile(inputDir + "/reason.dat").map(_.split('|')).filter(p => judge(p, "reason")).map(p =>
      Reason(p(0).trim.toLong, p(1).trim, p(2).trim)).toDF(),

    "ship_mode" -> sc.textFile(inputDir + "/ship_mode.dat").map(_.split('|')).filter(p => judge(p, "ship_mode")).map(p =>
      Ship_mode(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim, p(5).trim)).toDF(),

    "store" -> sc.textFile(inputDir + "/store.dat").map(_.split('|')).filter(p => judge(p, "store")).map(p =>
      Store(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim.toLong, p(5).trim, p(6).trim.toLong, p(7).trim.toLong, p(8).trim, p(9).trim, p(10).trim.toLong, p(11).trim, p(12).trim, p(13).trim, p(14).trim.toLong, p(15).trim, p(16).trim.toLong, p(17).trim, p(18).trim, p(19).trim, p(20).trim, p(21).trim, p(22).trim, p(23).trim, p(24).trim, p(25).trim, p(26).trim, p(27).trim.toDouble, p(28).trim.toDouble)).toDF(),

    "time_dim" -> sc.textFile(inputDir + "/time_dim.dat").map(_.split('|')).filter(p => judge(p, "time_dim")).map(p =>
      Time_dim(p(0).trim.toLong, p(1).trim, p(2).trim.toLong, p(3).trim.toLong, p(4).trim.toLong, p(5).trim.toLong, p(6).trim, p(7).trim, p(8).trim, p(9).trim)).toDF(),

    "warehouse" -> sc.textFile(inputDir + "/warehouse.dat").map(_.split('|')).filter(p => judge(p, "warehouse")).map(p =>
      Warehouse(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim.toLong, p(4).trim, p(5).trim, p(6).trim, p(7).trim, p(8).trim, p(9).trim, p(10).trim, p(11).trim, p(12).trim, p(13).trim.toDouble)).toDF(),

    "web_page" -> sc.textFile(inputDir + "/web_page.dat").map(_.split('|')).filter(p => judge(p, "web_page")).map(p =>
      Web_page(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim.toLong, p(5).trim.toLong, p(6).trim, p(7).trim.toLong, p(8).trim, p(9).trim, p(10).trim.toLong, p(11).trim.toLong, p(12).trim.toLong, p(13).trim.toLong)).toDF(),

    "web_site" -> sc.textFile(inputDir + "/web_site.dat").map(_.split('|')).filter(p => judge(p, "web_site")).map(p =>
      Web_site(p(0).trim.toLong, p(1).trim, p(2).trim, p(3).trim, p(4).trim, p(5).trim.toLong, p(6).trim.toLong, p(7).trim, p(8).trim, p(9).trim.toLong, p(10).trim, p(11).trim, p(12).trim, p(13).trim.toLong, p(14).trim, p(15).trim, p(16).trim, p(17).trim, p(18).trim, p(19).trim, p(20).trim, p(21).trim, p(22).trim, p(23).trim, p(24).trim, p(25).trim.toDouble)).toDF()
  )
  // for implicits
  val catalog_sales = dfMap.get("catalog_sales").get
  val catalog_returns = dfMap.get("catalog_returns").get
  val inventory = dfMap.get("inventory").get
  val store_sales = dfMap.get("store_sales").get
  val store_returns = dfMap.get("store_returns").get
  val web_sales = dfMap.get("web_sales").get
  val web_returns = dfMap.get("web_returns").get
  val call_center = dfMap.get("call_center").get
  val catalog_page = dfMap.get("catalog_page").get
  val customer = dfMap.get("customer").get
  val customer_address = dfMap.get("customer_address").get
  val customer_demographics = dfMap.get("customer_demographics").get
  val date_dim = dfMap.get("date_dim").get
  val household_demographics = dfMap.get("household_demographics").get
  val income_band = dfMap.get("income_band").get
  val item = dfMap.get("item").get
  val promotion = dfMap.get("promotion").get
  val reason = dfMap.get("reason").get
  val ship_mode = dfMap.get("ship_mode").get
  val store = dfMap.get("store").get
  val time_dim = dfMap.get("time_dim").get
  val warehouse = dfMap.get("warehouse").get
  val web_page = dfMap.get("web_page").get
  val web_site = dfMap.get("web_site").get

  val table_list = Seq( catalog_sales, catalog_returns, inventory, store_sales, store_returns, web_sales, web_returns, call_center, catalog_page, customer, customer_address, customer_demographics, date_dim, household_demographics, income_band, item, promotion, reason, ship_mode, store, time_dim, warehouse, web_page, web_site )

  val table_names = Seq( "catalog_sales", "catalog_returns", "inventory", "store_sales", "store_returns", "web_sales", "web_returns", "call_center", "catalog_page", "customer", "customer_address", "customer_demographics", "date_dim", "household_demographics", "income_band", "item", "promotion", "reason", "ship_mode", "store", "time_dim", "warehouse", "web_page", "web_site" )

  dfMap.foreach {
    case (key, value) => value.createOrReplaceTempView(key)
  }
}
